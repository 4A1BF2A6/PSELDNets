# src/models/components/moa_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import math

class CosineTopKGate(nn.Module):
    """
    基于余弦相似度的Top-K门控机制，用于选择专家。
    源自 MoA 项目。
    """
    def __init__(self, model_dim, num_global_experts, k=1, fp32_gate=False, proj_dim=256, init_t=0.5, **options):
        super(CosineTopKGate, self).__init__()
        # 选择top-k个专家，k=1时，选择最优专家，取全局专家和k中的较小值
        self.top_k = min(num_global_experts, int(k))
        self.fp32_gate = fp32_gate # 是否使用FP32进行门控计算
        # 可学习的温度参数，用于调整logits的缩放
        self.temperature = nn.Parameter(torch.log(torch.full([1], 1.0 / init_t)), requires_grad=True)
        # 线性投影层，用于将输入特征投影到较低维度
        self.cosine_projector = nn.Linear(model_dim, proj_dim)
        # 相似性矩阵，每列代表一个专家的原型向量
        self.sim_matrix = nn.Parameter(torch.randn(size=(proj_dim, num_global_experts)), requires_grad=True)
        # 温度参数的最大钳位值，防止过大，数值不稳定
        self.clamp_max = torch.log(torch.tensor(1.0 / 0.01)).item() 
        # 初始化相似性矩阵
        torch.nn.init.normal_(self.sim_matrix, 0, 0.01)

        # 检查是否有未识别的选项
        for opt in options:
            if opt not in ('capacity_factor', 'gate_noise'):
                raise Exception('Unrecognized argument provided to Gating module: %s' % opt)

    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 [batch_size * seq_len, model_dim]
        Returns:
            logits: 每个专家对应的logit值，形状为 [batch_size * seq_len, num_global_experts]
        """
        # 根据fp32_gate设置，选择是否将计算转换为float32类型
        if self.fp32_gate:
            x_proc = x.float()
            cosine_projector_proc = self.cosine_projector.float()
            sim_matrix_proc = self.sim_matrix.float()
        else:
            x_proc = x
            cosine_projector_proc = self.cosine_projector
            sim_matrix_proc = self.sim_matrix
        
        # 将输入投影到较低维度并归一化
        projected_x = F.normalize(cosine_projector_proc(x_proc), dim=1)
        # 归一化专家原型向量
        normalized_sim_matrix = F.normalize(sim_matrix_proc, dim=0)
        
        # 计算投影后的输入与专家原型之间的余弦相似度
        logits = torch.matmul(projected_x, normalized_sim_matrix)
        
        # 应用温度缩放
        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        logits = logits * logit_scale
        return logits

def load_importance_loss(scores_wo_noise, topk_logits, num_global_experts, gate_noise):
    """
    计算负载均衡损失（Load Balancing Loss）和重要性损失（Importance Loss）。
    源自 MoA 项目。
    
    Args:
        scores_wo_noise: 未加噪声的原始分数 (softmax输出)
        topk_logits: 每个输入选择的top-k专家的logit值
        num_global_experts: 专家总数
        gate_noise: 门控噪声的缩放因子
        
    Returns:
        总的辅助损失
    """
    
    # 重要性损失: 衡量所有专家被选择的均衡程度
    # 希望所有专家都被差不多地使用
    def importance_loss_fn(scores):
        # 计算每个专家被选择的总分数
        expert_importance = scores.float().sum(0) 
        # 计算重要性分数的变异系数的平方，作为损失
        # (var / mean^2)
        loss = expert_importance.float().var() / (expert_importance.float().mean() ** 2 + 1e-10)
        return loss

    # 负载损失: 衡量每个专家处理的负载均衡程度
    # 基于专家选择的概率和阈值（top-k中最低的logit）
    def load_loss_fn(scores, top_logits):
        assert gate_noise > 0, "`gate_noise` must be > 0 for normalization in load_importance_loss()."
        # 使用正态分布的累积分布函数 (CDF) 来估计负载
        # 假设门控的噪声服从正态分布
        normal_dist = Normal(
            torch.tensor([0.0], device=scores.device),
            torch.tensor([gate_noise / num_global_experts], device=scores.device),
        )
        # 将top-k中最低的logit作为阈值
        threshold = top_logits[:, -1].view(-1, 1).float() 
        # 计算原始分数与阈值之差
        diff = scores.float() - threshold.float()
        # 使用CDF估计每个专家被选中的概率（如果分数高于阈值）
        prob = normal_dist.cdf(diff)
        # 计算每个专家的总负载
        total_load_per_expert = prob.sum(0)
        # 计算负载的变异系数的平方，作为损失
        loss = total_load_per_expert.float().var() / (total_load_per_expert.float().mean() ** 2 + 1e-10)
        return loss

    # 计算两种损失
    l_imp = importance_loss_fn(scores_wo_noise)
    l_load = load_loss_fn(scores_wo_noise, topk_logits)
    
    # 返回两种损失的平均值，并乘以一个小的系数（如0.01）进行缩放
    return (l_imp + l_load) / 2.0