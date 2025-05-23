# src/models/components/mixture_of_existing_adapters.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .moa_utils import CosineTopKGate, load_importance_loss
from .model_utilities_adapt import Adapter, DCTAdapter, DCTFrequencyAdapter, SEAdapter

class MixtureOfExistingAdapters(nn.Module):
    """
    混合适配器模块，支持动态添加不同类型的专家适配器
    """
    def __init__(self, in_features,
                 experts_config=None,  # 新参数：专家配置列表
                 dct_adapter_kwargs=None, 
                 freq_adapter_kwargs=None,
                 adapter_kwargs=None,   # 普通适配器参数
                 router_kwargs=None,
                 gate_noise_factor=1.0, 
                 aux_loss_coeff=0.01,
                 **kwargs):
        super().__init__()
        self.in_features = in_features
        self.gate_noise_factor = gate_noise_factor
        self.aux_loss_coeff = aux_loss_coeff
        
        # 构建专家列表
        self.experts = nn.ModuleList()
        self.expert_names = []
        
        # 将 OmegaConf 的 ListConfig 转换为 Python 原生列表
        experts_config = list(experts_config)
        
        # 处理专家配置
        if experts_config and isinstance(experts_config, list):
            # 如果提供了专家配置列表，则使用它来创建专家
            for config in experts_config:
                expert_type = config.get('type', 'adapter')
                expert_name = config.get('name', f'expert_{len(self.experts)}')
                expert_kwargs = config.get('kwargs', {})
                
                # 创建专家并添加到列表中
                expert = self._create_expert(expert_type, in_features, expert_kwargs)
                self.experts.append(expert)
                self.expert_names.append(expert_name)
                print(f"添加专家 {expert_name} (类型: {expert_type})")
        else:
            # 为了向后兼容，如果没有提供专家配置，则使用旧方式创建专家
            if dct_adapter_kwargs:
                self.experts.append(DCTAdapter(in_features=in_features, **(dct_adapter_kwargs or {})))
                self.expert_names.append('dct_expert')
                print("添加 DCT 专家")
            
            if freq_adapter_kwargs:
                self.experts.append(DCTFrequencyAdapter(in_features=in_features, **(freq_adapter_kwargs or {})))
                self.expert_names.append('freq_expert')
                print("添加 Frequency 专家")
                
            if adapter_kwargs:
                self.experts.append(Adapter(in_features=in_features, **(adapter_kwargs or {})))
                self.expert_names.append('adapter_expert')
                print("添加普通 Adapter 专家")
        
        # 确保至少有一个专家
        if not self.experts:
            # 如果没有专家，添加一个默认的普通适配器作为专家
            self.experts.append(Adapter(in_features=in_features))
            self.expert_names.append('default_expert')
            print("添加默认专家")
            
        # 专家数量
        self.num_experts = len(self.experts)
        print(f"总共添加了 {self.num_experts} 个专家")
        
        # 实例化路由器
        _router_kwargs = router_kwargs if router_kwargs else {}
        self.router = CosineTopKGate(model_dim=in_features, num_global_experts=self.num_experts, **_router_kwargs)
        
        # 层归一化
        self.norm = nn.LayerNorm(in_features)
        self.aux_loss = torch.tensor(0.0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    def _create_expert(self, expert_type, in_features, kwargs):
        """
        根据专家类型创建相应的专家实例
        
        Args:
            expert_type (str): 专家类型，可选值: 'dct', 'frequency', 'adapter'等
            in_features (int): 输入特征维度
            kwargs (dict): 专家的额外参数
            
        Returns:
            nn.Module: 创建的专家模块
        """
        if expert_type == 'dct':
            return DCTAdapter(in_features=in_features, **kwargs)
        elif expert_type == 'frequency':
            return DCTFrequencyAdapter(in_features=in_features, **kwargs)
        elif expert_type == 'adapter':
            return Adapter(in_features=in_features, **kwargs)
        elif expert_type == 'SEAdapter':
            return SEAdapter(in_features=in_features, **kwargs)
        else:
            raise ValueError(f"不支持的专家类型: {expert_type}")

    def forward(self, x, residual_input=None):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, in_features]
            residual_input (torch.Tensor, optional): 用于残差连接的输入。如果为None，则使用 x。
        Returns:
            torch.Tensor: 处理后的张量
        """
        if residual_input is None:
            residual_input = x

        # 1. 层归一化
        x_normalized = self.norm(x)
        
        # 2. 路由
        batch_size, seq_len, _ = x_normalized.shape
        # 将输入展平以适应路由器
        x_flat = rearrange(x_normalized, 'b s d -> (b s) d') 
        
        # 获取路由器的原始logits
        router_logits_no_noise = self.router(x_flat) # shape: [batch_size * seq_len, num_experts]
        
        # 在训练时添加噪声以鼓励探索
        if self.training and self.gate_noise_factor > 0:
            noise = torch.randn_like(router_logits_no_noise) * self.gate_noise_factor / self.num_experts
            router_logits = router_logits_no_noise + noise
        else:
            router_logits = router_logits_no_noise
        
        # 选择top-k专家
        if self.router.top_k > 0:
            # 获取top-k的logits和索引
            topk_logits, topk_indices = torch.topk(router_logits, k=self.router.top_k, dim=1)
            # 创建one-hot选择矩阵
            one_hot = torch.zeros_like(router_logits).scatter_(1, topk_indices, 1.0)
            # 计算softmax权重
            routing_weights = F.softmax(topk_logits, dim=1)
            # 将routing_weights扩展到与router_logits相同的维度
            expanded_weights = torch.zeros_like(router_logits)
            expanded_weights.scatter_(1, topk_indices, routing_weights)
            # 只保留top-k的权重,其他置为0
            routing_weights = expanded_weights * one_hot
        else:
            # 使用所有专家的softmax权重
            routing_weights = F.softmax(router_logits, dim=1)
        
        # 重新整形路由权重
        effective_routing_weights = rearrange(routing_weights, '(b s) e -> b s e', b=batch_size) # [B, S, num_experts]

        # 3. 应用专家并收集输出
        expert_outputs = []
        for expert in self.experts:
            # 对每个专家，使用归一化后的输入 x_normalized
            expert_output = expert(x_normalized) # [B, S, D]
            expert_outputs.append(expert_output)
        
        # 将专家输出堆叠起来
        expert_outputs = torch.stack(expert_outputs, dim=2) # [B, S, num_experts, D]
        
        # 4. 根据路由权重混合专家输出
        weighted_output = torch.sum(effective_routing_weights.unsqueeze(-1) * expert_outputs, dim=2) # [B, S, D]
        
        # # 5. 计算辅助损失 (仅在训练时)
        # if self.training:
        #     softmax_scores_no_noise = F.softmax(router_logits_no_noise, dim=1)
        #     aux_loss = self.aux_loss_coeff * load_importance_loss(
        #         softmax_scores_no_noise, 
        #         router_logits_no_noise,
        #         self.num_experts, 
        #         self.gate_noise_factor
        #     )
        #     self.aux_loss = aux_loss
            
        #     # 添加到全局辅助损失列表，以便在训练步骤中收集
        #     if not hasattr(torch, 'global_aux_loss'):
        #         torch.global_aux_loss = []
        #     torch.global_aux_loss.append(aux_loss)
        # else:
        #     self.aux_loss = torch.tensor(0.0, device=x.device)

        # 6. 返回适配器效果 (delta)
        return weighted_output, self.aux_loss