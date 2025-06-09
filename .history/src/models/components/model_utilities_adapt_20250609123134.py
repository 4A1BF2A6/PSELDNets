import math
import torch
import torch_dct as dct
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Optional
from torch import Tensor
import numpy as np

class Adapter(nn.Module):
    def __init__(self, in_features, mlp_ratio=0.5, act_layer='gelu', 
                 adapter_scalar=0.1, **kwargs):
        super().__init__()
        hidden_features = int(in_features * mlp_ratio)
        if act_layer == 'gelu':
            self.act = nn.GELU()
        elif act_layer == 'relu':
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Activation layer {act_layer} not supported")
        if adapter_scalar == 'learnable_scalar':
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = adapter_scalar
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)

        self.init_weights()

    def init_weights(self):
        # NOTE this init overrides that base model init with specific changes for the block type
        # if self.init_values is not None:
        nn.init.constant_(self.fc2.weight, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x, residual=None):
        # x.shape is [num_windows*B, N, C] == x_main.shape is [num_windows*B, token的数量, 每个补丁的特征维度]
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x * self.scale
        if residual is not None:
            x = x + residual

        return x


class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
    

class Conv2d(nn.Conv2d, LoRALayer):
    def __init__(self, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, **kwargs):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0., merge_weights=False)
        # Actual trainable parameters
        if r > 0:
            stride = self.stride
            padding = self.padding
            self.lora_A = nn.Conv2d(in_channels, r, kernel_size, stride=stride, padding=padding, bias=False)
            self.lora_B = nn.Conv2d(r, out_channels, (1, 1), (1, 1), bias=False)
            print(self.lora_A.weight.shape, self.lora_B.weight.shape, in_channels, r, out_channels)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        if self.r > 0:
            result = nn.Conv2d.forward(self, x)
            result = result + self.lora_B(self.lora_A(x)) * self.scaling
        return result


# 在 src/models/components/model_utilities_adapt.py 添加DCT适配器类
class DCTAdapter(nn.Module):
    """离散余弦变换(DCT)增强的Adapter，专注于频率域信息处理
    
    Args:
        in_features (int): 输入特征维度
        mlp_ratio (float): 隐藏层维度与输入维度的比例
        act_layer (str): 激活函数类型，'gelu'或'relu'
        adapter_scalar (float或str): 适配器缩放因子，可以是浮点数或'learnable_scalar'
        dct_kernel_size (int): DCT卷积核大小
        dct_groups (int): DCT分组卷积的组数
        freq_dim (int): 频率维度，默认为-1表示特征向量的最后一个维度
    """
    def __init__(self, in_features, mlp_ratio=0.25, act_layer='gelu',
                 adapter_scalar=1, dct_kernel_size=3, dct_groups=1, 
                 freq_dim=-1, **kwargs):
        super().__init__()
        hidden_features = int(in_features * mlp_ratio)
        
        # 配置激活函数
        if act_layer == 'gelu':
            self.act = nn.GELU()
        elif act_layer == 'relu':
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Activation layer {act_layer} not supported")
            
        # 配置缩放因子
        if adapter_scalar == 'learnable_scalar':
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = adapter_scalar
        
        # DCT变换相关参数
        self.dct_kernel_size = dct_kernel_size
        self.dct_groups = dct_groups
        self.freq_dim = freq_dim
        
        # 生成DCT基函数
        self.register_buffer('dct_basis', self._get_dct_basis(dct_kernel_size))
        
        # 频域转换层
        self.freq_down = nn.Linear(max(12, self.dct_kernel_size), hidden_features)
        self.freq_up = nn.Linear(hidden_features, in_features)
        
        # 残差连接前的层归一化
        self.norm = nn.LayerNorm(in_features)
        
        self.init_weights()
        
    def _get_dct_basis(self, kernel_size):
        """生成DCT基函数矩阵"""
        dct_basis = torch.zeros(kernel_size, kernel_size)
        for k in range(kernel_size):
            for n in range(kernel_size):
                if k == 0:
                    dct_basis[k, n] = 1.0 / math.sqrt(kernel_size)
                else:
                    dct_basis[k, n] = math.sqrt(2.0 / kernel_size) * math.cos(
                        math.pi * (n + 0.5) * k / kernel_size)
        return dct_basis
    
    def init_weights(self):
        """初始化权重，使输出接近零"""
        nn.init.kaiming_uniform_(self.freq_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.freq_down.bias)
        nn.init.zeros_(self.freq_up.weight)
        nn.init.zeros_(self.freq_up.bias)
    
    def apply_dct_conv(self, x):
        """应用DCT卷积操作提取频率特征
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, in_features]
            
        Returns:
            频率增强的特征张量
        """
        # 手动实现DCT卷积
        # batch_size, seq_len, dim = x.shape
        
        # # 为了更好地处理边界，我们使用反射填充
        # padding = self.dct_kernel_size // 2
        
        # # 对频率维度应用DCT卷积
        # # 首先重塑为形状便于卷积操作
        # x_reshaped = x.transpose(1, 2).contiguous()  # [B, D, L]
        
        # # 应用反射填充
        # x_padded = F.pad(x_reshaped, (padding, padding), mode='reflect')
        
        # # 提取滑动窗口
        # x_windows = []
        # for i in range(seq_len):
        #     window = x_padded[:, :, i:i+self.dct_kernel_size]
        #     x_windows.append(window)
        
        # # 将窗口堆叠成批次
        # x_windows = torch.stack(x_windows, dim=2)  # [B, D, L, K]
        
        # # 应用DCT变换
        # dct_features = torch.matmul(x_windows, self.dct_basis.t())  # [B, D, L, K]
        
        # # 只保留低频分量（前几个DCT系数）
        # keep_freqs = max(1, self.dct_kernel_size // 2)
        # dct_features = dct_features[:, :, :, :keep_freqs]
        
        # # 重新展平
        # dct_features = dct_features.reshape(batch_size, dim, seq_len, -1)
        # dct_features = dct_features.permute(0, 2, 1, 3).contiguous()
        # dct_features = dct_features.reshape(batch_size, seq_len, -1)

        # 使用torch_dct实现DCT卷积 x.shape is [B, N, C]
        # x_transposed = x.transpose(1, 2)  # -> [B, C, N]
    
        # 在通道维度上做 DCT
        dct_output = dct.dct(x, norm='ortho')  # DCT-II with orthogonal normalization
        
        # 只保留低频分量（前几个DCT系数）
        keep_freqs = max(12, self.dct_kernel_size)
        dct_output = dct_output[:, :, :keep_freqs]  # -> [B, N, keep_freqs]

        return dct_output
    
    def forward(self, x, residual=None):
        """前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, in_features]
            residual: 可选的残差连接输入
            
        Returns:
            处理后的张量
        """
        # 1. 应用层归一化
        x_norm = self.norm(x)
        
        # 2. 提取DCT频率特征
        dct_features = self.apply_dct_conv(x_norm)
        
        # 3. 通过MLP处理DCT特征
        hidden = self.freq_down(dct_features)
        hidden = self.act(hidden)
        out = self.freq_up(hidden)
        
        # 4. 应用缩放
        out = out * self.scale
        
        # 5. 加上残差连接(如果提供)
        if residual is not None:
            out = out + residual
        
        return out

class DCTFrequencyAdapter(nn.Module):
    """
        简化版频率适配器，避免维度问题
    """

    def __init__(self, in_features, mlp_ratio=0.25, act_layer='gelu',
                 adapter_scalar=1, **kwargs):
        super().__init__()
        hidden_features = int(in_features * mlp_ratio)
        
        # 基础适配器组件
        if act_layer == 'gelu':
            self.act = nn.GELU()
        elif act_layer == 'relu':
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Activation layer {act_layer} not supported")
            
        if adapter_scalar == 'learnable_scalar':
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = adapter_scalar
        
        # 频率通道注意力
        self.channel_attention = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, in_features),
            nn.Sigmoid()
        )
        
        # MLP路径
        self.norm = nn.LayerNorm(in_features)
        self.down = nn.Linear(in_features, hidden_features)
        self.up = nn.Linear(hidden_features, in_features)

    def forward(self, x, residual=None):

        '''
            x.shape is [B, L, D]
            for starss23, L = 1000, D = 128
            
        '''
        # 应用层归一化
        x_norm = self.norm(x)
        
        # 计算通道注意力
        attn = self.channel_attention(x_norm)
        
        # 应用通道注意力
        x_attn = x_norm * attn
        
        # MLP处理
        hidden = self.down(x_attn)
        hidden = self.act(hidden)
        output = self.up(hidden)
        
        # 缩放
        output = output * self.scale
        
        # 可选残差
        if residual is not None:
            output = output + residual
            
        return output

class SEAdapter(nn.Module):
    """
        SE适配器，通道注意力
        传统的SE模型，[B, C, H, W]
        我的特征形状是 [B, N, C]
    """

    def __init__(self, in_features, mlp_ratio=0.25, act_layer='gelu',
                 adapter_scalar=1, **kwargs):
        super().__init__()
        hidden_features = int(in_features * mlp_ratio)
        
        # 基础适配器组件
        if act_layer == 'gelu':
            self.act = nn.GELU()
        elif act_layer == 'relu':
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Activation layer {act_layer} not supported")
            
        if adapter_scalar == 'learnable_scalar':
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = adapter_scalar
        
        self.norm = nn.LayerNorm(in_features)

        # 传统SE模块组件
        self.globalAvgPool = nn.AdaptiveAvgPool1d(1)
        self.down = nn.Linear(in_features, hidden_features)
        self.up = nn.Linear(hidden_features, in_features)
        self.sigmoid = nn.Sigmoid()

    def channelAttention(self, x):
        '''
            x.shape is [B, C, N]
        '''
        # 计算通道注意力
        out = self.globalAvgPool(x) # [B, C, 1]
        out = self.down(out.transpose(1, 2)) # [B, 1, C/r]
        out = self.act(out) # [B, 1, C/r]
        out = self.up(out) # [B, 1, C]
        attn = self.sigmoid(out) # [B, 1, C]

        return attn
       

    def forward(self, x, residual=None):

        '''
            x.shape is [B(num_windows*B), N(token的数量), C(每个补丁的特征维度)]
        '''

        # 应用层归一化
        # x_norm = self.norm(x)

        x = self.down(x)
        x = self.act(x)
        x = self.up(x)


        # 转置以适应传统SE模块 [B, N, C] -> [B, C, N]
        x_t = x.transpose(1, 2)
        
        # 计算通道注意力
        # out = self.globalAvgPool(x_t) # [B, C, 1]
        # out = self.down(out.transpose(1, 2)) # [B, 1, C/r]
        # out = self.act(out) # [B, 1, C/r]
        # out = self.up(out) # [B, 1, C]
        # attn = self.sigmoid(out) # [B, 1, C]
        attn = self.channelAttention(x_t)
        
        # 应用通道注意力
        x_attn = x * attn # [B, N, C]

        output = x_attn
        # 再通过一个MLP
        # output = self.down(output)
        # output = self.act(output)
        # output = self.up(output)

        # 缩放
        output = output * self.scale
        
        # 可选残差
        if residual is not None:
            output = output + residual
            
        return output
    

class SqueezeExcitation(torch.nn.Module):
    """
    实现Squeeze-and-Excitation模块，来自论文 https://arxiv.org/abs/1709.01507
    参数说明：
        input_channels (int): 输入图像的通道数
        squeeze_channels (int): 压缩后的通道数
        activation (Callable): delta激活函数，默认为ReLU
        scale_activation (Callable): sigma激活函数，默认为Sigmoid
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        out_channels: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        # 全局平均池化
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        # 第一个全连接层，用于降维
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        # 第二个全连接层，用于升维
        self.fc2 = torch.nn.Conv2d(squeeze_channels, out_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        """计算通道注意力权重"""
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        """前向传播"""
        scale = self._scale(input)
        # return scale * input  # 原始SE模块会返回加权后的特征
        return scale  # 这里只返回注意力权重


class ConvAdapterDesign1(nn.Module):
    """
    Conv-Adapter的第一个设计方案
    结构：1x1卷积 -> 深度卷积 -> 1x1卷积
    输入形状: [B, N, C] -> [B, C, N, 1] -> [B, N, C]
    """
    def __init__(self, in_features, mlp_ratio=0.25, act_layer='gelu',
                 adapter_scalar=1, kernel_size=3, padding=1, stride=1, 
                 groups=1, dilation=1, **kwargs):
        super().__init__()

        # 计算隐藏层维度
        hidden_features = int(in_features * mlp_ratio)
        
        # 配置激活函数
        if act_layer == 'gelu':
            self.act = nn.GELU()
        elif act_layer == 'relu':
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Activation layer {act_layer} not supported")
            
        # 配置缩放因子
        if adapter_scalar == 'learnable_scalar':
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = adapter_scalar

        # 1x1点卷积，用于降维
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.norm1 = nn.LayerNorm([hidden_features])

        # 深度卷积，用于特征提取
        self.conv2 = nn.Conv2d(hidden_features, hidden_features, 
                              kernel_size=kernel_size, stride=stride, 
                              groups=groups, padding=padding, 
                              dilation=int(dilation))
        self.norm2 = nn.LayerNorm([hidden_features])

        # 1x1点卷积，用于升维
        self.conv3 = nn.Conv2d(hidden_features, in_features, kernel_size=1, stride=1)
        self.norm3 = nn.LayerNorm([in_features])
    
    def forward(self, x, residual=None):
        """
        前向传播
        Args:
            x: 输入张量，形状为 [B, N, C]
            residual: 可选的残差连接输入
        Returns:
            处理后的张量，形状为 [B, N, C]
        """
        # 检查输入维度
        if len(x.shape) == 3:  # [B, N, C]

            # 保存原始形状
            B, N, C = x.shape
            
            # 重塑输入以适应卷积层 [B, N, C] -> [B, C, N, 1]
            x = x.transpose(1, 2).unsqueeze(-1)
            
            # 第一个1x1卷积
            out = self.conv1(x)  # [B, hidden_features, N, 1]
            out = out.squeeze(-1).transpose(1, 2)  # [B, N, hidden_features]
            out = self.norm1(out)
            out = self.act(out)
            out = out.transpose(1, 2).unsqueeze(-1)  # [B, hidden_features, N, 1]
            
            # 深度卷积
            out = self.conv2(out)  # [B, hidden_features, N, 1]
            out = out.squeeze(-1).transpose(1, 2)  # [B, N, hidden_features]
            out = self.norm2(out)
            out = self.act(out)
            out = out.transpose(1, 2).unsqueeze(-1)  # [B, hidden_features, N, 1]

            # 第二个1x1卷积
            out = self.conv3(out)  # [B, in_features, N, 1]
            out = out.squeeze(-1).transpose(1, 2)  # [B, N, in_features]
            out = self.norm3(out)
            out = self.act(out)
            # 应用缩放
            out = out * self.scale
        
            # 可选残差连接
            if residual is not None:
                out = out + residual

        else:  # 四维输入 [B, C, H, W]
            B, C, H, W = x.shape
            # 保持原始形状

            # 第一个1x1卷积
            out = self.conv1(x)
            # out.shape is [B, C, H, W]  norm accept shape is B H W C
            out = self.norm1(out.permute(0, 2, 3, 1))
            out = self.act(out).permute(0, 3, 1, 2) # [B, C, H, W]
            
            # 深度卷积
            out = self.conv2(out)
            out = self.norm2(out.permute(0, 2, 3, 1)) # [B, H, W, C]
            out = self.act(out).permute(0, 3, 1, 2) # [B, C, H, W]

            # 第二个1x1卷积
            out = self.conv3(out)
            out = self.norm3(out.permute(0, 2, 3, 1)) # [B, H, W, C]
            out = self.act(out).permute(0, 3, 1, 2) # [B, C, H, W]

            # 应用缩放
            out = out * self.scale
            
            out = out.reshape(B, H*W, C)
            # 可选残差连接
            if residual is not None:
                out = out + residual

        return out


class ConvAdapter(nn.Module):
    """
    Conv-Adapter的第二个设计方案（v4版本）
    结构：深度卷积 -> 1x1卷积 + 通道注意力
    """
    def __init__(self, inplanes, outplanes, width, 
                kernel_size=3, padding=1, stride=1, groups=1, dilation=1, norm_layer=None, act_layer=None, **kwargs):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.Identity
        if act_layer is None:
            act_layer = nn.Identity

        # 深度卷积，用于特征提取
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=kernel_size, stride=stride, groups=groups, padding=padding, dilation=int(dilation))
        self.act = act_layer()

        # 1x1点卷积，用于通道调整
        self.conv2 = nn.Conv2d(width, outplanes, kernel_size=1, stride=1)

        # 通道注意力机制
        # 方案1：使用SE模块
        # self.se = SqueezeExcitation(inplanes, width, outplanes, activation=act_layer)
        # 方案2：使用可学习的通道缩放参数
        self.se = nn.Parameter(1.0 * torch.ones((1, outplanes, 1, 1)), requires_grad=True)
    
    def forward(self, x):
        """前向传播"""
        # 深度卷积
        out = self.conv1(x)
        out = self.act(out)
        # 1x1卷积
        out = self.conv2(out)
        # 通道注意力
        out = out * self.se
        return out


class LinearAdapter(nn.Module):
    """
    线性适配器模块
    用于处理一维特征，结构：线性层 -> 激活 -> 线性层 + 通道注意力
    """
    def __init__(self, inplanes, outplanes, width, act_layer=None, **kwargs):
        super().__init__()

        # 第一个线性层，用于降维
        self.fc1 = nn.Linear(inplanes, width)
        # 第二个线性层，用于升维
        self.fc2 = nn.Linear(width, outplanes)
        self.act = act_layer()
        # 通道注意力参数
        self.se = nn.Parameter(1.0 * torch.ones((1, outplanes)), requires_grad=True)

    def forward(self, x):
        """前向传播"""
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        out = out * self.se
        return out
    
'''
    
    原论文里面的加权卷积模块

'''
class wConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, den, stride=1, padding=1, groups=1, bias=False):
        super(wConv2d, self).__init__()       
        # 初始化基本参数
        self.stride = stride          # 卷积步长
        self.padding = padding        # 填充大小
        self.kernel_size = kernel_size # 卷积核大小
        self.groups = groups          # 分组卷积的组数
        
        # 初始化卷积权重
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        # 使用Kaiming初始化方法初始化权重
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')        
        
        # 初始化偏置项（如果bias=True）
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # 设置设备为CPU
        device = torch.device('cpu')  
        
        # 创建权重矩阵alfa
        # 将den数组与1.0和den的翻转版本拼接
        self.register_buffer('alfa', torch.cat([torch.tensor(den, device=device),
                                              torch.tensor([1.0], device=device),
                                              torch.flip(torch.tensor(den, device=device), dims=[0])]))
        
        # 计算外积矩阵Phi
        self.register_buffer('Phi', torch.outer(self.alfa, self.alfa))

        # 检查Phi矩阵的维度是否与卷积核大小匹配
        if self.Phi.shape != (kernel_size, kernel_size):
            raise ValueError(f"Phi shape {self.Phi.shape} must match kernel size ({kernel_size}, {kernel_size})")

    def forward(self, x):
        # 将Phi矩阵移动到与输入相同的设备上
        Phi = self.Phi.to(x.device)
        # 将权重与Phi矩阵相乘
        weight_Phi = self.weight * Phi
        # 执行卷积操作
        return F.conv2d(x, weight_Phi, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)

'''
    我的加权卷积模块
'''
class WConvAdapter(nn.Module):
    """
    Conv-Adapter的第一个设计方案（v2版本）
    结构：1x1卷积 -> 加权深度卷积 -> 1x1卷积 
    输入形状: [B, C, H, W] -> [B, C, H, W]
    """
    def __init__(self, in_features, mlp_ratio=0.5, act_layer='gelu',
                 adapter_scalar=1, kernel_size=3, padding=1, stride=1, 
                 groups=1, dilation=1, den=None, **kwargs):
        super().__init__()

        # 计算隐藏层维度
        hidden_features = int(in_features * mlp_ratio)
        
        # 配置激活函数
        if isinstance(act_layer, str):
            if act_layer.lower() == 'gelu':
                self.act = nn.GELU()
            elif act_layer.lower() == 'relu':
                self.act = nn.ReLU()
            else:
                raise ValueError(f"Activation layer {act_layer} not supported")
        else:
            self.act = act_layer()
            
        # 配置缩放因子
        if adapter_scalar == 'learnable_scalar':
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = adapter_scalar

        # 设置默认的den参数
        if den is None:
            den = [0.7, 1.0, 0.7]  # 默认的3x3卷积核权重

        # 1x1点卷积，用于降维
        self.conv1 = wConv2d(in_features, hidden_features, 
                            kernel_size=1, stride=1
                            )  # 1x1卷积使用单一权重
        self.norm1 = nn.LayerNorm([hidden_features])

        # 深度卷积，用于特征提取
        self.conv2 = wConv2d(hidden_features, hidden_features, 
                            kernel_size=kernel_size, stride=stride, 
                            groups=groups, padding=padding, 
                            den=den)  # 使用传入的den参数
        self.norm2 = nn.LayerNorm([hidden_features])

        # 1x1点卷积，用于升维
        self.conv3 = wConv2d(hidden_features, in_features, 
                            kernel_size=1, stride=1
                            )  # 1x1卷积使用单一权重
        self.norm3 = nn.LayerNorm([in_features])
    
    def forward(self, x, residual=None):
        """
        前向传播
        Args:
            x: 输入张量，形状为 [B, C, H, W]
        Returns:
            处理后的张量，形状为 [B, C, H, W]
        """
        # 检查输入维度
        if len(x.shape) == 3:  # [B, N, C]
            # 保存原始形状
            B, N, C = x.shape
            
            # 重塑输入以适应卷积层 [B, N, C] -> [B, C, N, 1]
            x = x.transpose(1, 2).unsqueeze(-1)
            
            # 第一个1x1卷积
            out = self.conv1(x)  # [B, hidden_features, N, 1]
            out = out.squeeze(-1).transpose(1, 2)  # [B, N, hidden_features]
            out = self.norm1(out)
            out = self.act(out)
            out = out.transpose(1, 2).unsqueeze(-1)  # [B, hidden_features, N, 1]
            
            # 深度卷积
            out = self.conv2(out)  # [B, hidden_features, N, 1]
            out = out.squeeze(-1).transpose(1, 2)  # [B, N, hidden_features]
            out = self.norm2(out)
            out = self.act(out)
            out = out.transpose(1, 2).unsqueeze(-1)  # [B, hidden_features, N, 1]

            # 第二个1x1卷积
            out = self.conv3(out)  # [B, in_features, N, 1]
            out = out.squeeze(-1).transpose(1, 2)  # [B, N, in_features]
            out = self.norm3(out)
            out = self.act(out)
            # 应用缩放
            out = out * self.scale
        
            # 可选残差连接
            if residual is not None:
                out = out + residual

        else:  # 四维输入 [B, C, H, W]
            B, C, H, W = x.shape
            # 保持原始形状

            # 第一个1x1卷积
            out = self.conv1(x)
            out = self.norm1(out.permute(0, 2, 3, 1))
            out = self.act(out).permute(0, 3, 1, 2)
            
            # 深度卷积
            out = self.conv2(out)
            out = self.norm2(out.permute(0, 2, 3, 1))
            out = self.act(out).permute(0, 3, 1, 2)

            # 第二个1x1卷积
            out = self.conv3(out)
            out = self.norm3(out.permute(0, 2, 3, 1))
            out = self.act(out).permute(0, 3, 1, 2)

            # 应用缩放
            out = out * self.scale
            
            out = out.reshape(B, H*W, C)
            # 可选残差连接
            if residual is not None:
                out = out + residual
        
        return out
    
# class WConvAdapterv2(nn.Module):
#     """
#     Conv-Adapter的第二个设计方案（v4版本）
#     结构：加权深度卷积 -> 1x1卷积 + 通道注意力
#     输入形状: [B, C, H, W] -> [B, C, H, W]
#     """
#     def __init__(self, inplanes, outplanes, width, 
#                 kernel_size=3, padding=1, stride=1, groups=1,
#                 act_layer=None, den=None, **kwargs):
#         super().__init__()

#         # 配置激活函数
#         if act_layer == 'gelu':
#             self.act = nn.GELU()
#         elif act_layer == 'relu':
#             self.act = nn.ReLU()
#         else:
#             raise ValueError(f"Activation layer {act_layer} not supported")

#         # if norm_layer is None:
#         #     norm_layer = nn.Identity
#         # if act_layer is None:
#         #     act_layer = nn.Identity
            
#         # SELD任务特定的权重设置
#         if den is None:
#             # 对于3x3卷积核，设置3个权重值
#             den = [0.7, 1.0, 0.7]  # 时间维度的权重分布

#         # 确保width能被groups整除
#         width = (width // groups) * groups

#         # 使用加权深度卷积
#         self.conv1 = wConv2d(inplanes, width, 
#                             kernel_size=kernel_size, 
#                             stride=stride, 
#                             groups=groups, 
#                             padding=padding, 
#                             den=den)

#         # 1x1点卷积
#         self.conv2 = nn.Conv2d(width, outplanes, kernel_size=1, stride=1, padding=0, groups=1)

#         # 通道注意力机制
#         self.se = nn.Parameter(1.0 * torch.ones((1, outplanes, 1, 1)), requires_grad=True)
    
#     def forward(self, x):
#         """
#         前向传播
#         Args:
#             x: 输入张量，形状为 [B, C, H, W]
#         Returns:
#             处理后的张量，形状为 [B, C, H, W]
#         """
#         # 加权深度卷积
#         out = self.conv1(x)
#         out = self.act(out)
        
#         # 1x1卷积
#         out = self.conv2(out)
        
#         # 通道注意力
#         out = out * self.se
        
#         return out


if __name__ == '__main__':
    # 测试代码
    # adapter = ConvAdapter(128, 128, width=32, groups=32)
    # print(adapter.conv1.weight.shape)  # 打印深度卷积权重形状
    # print(adapter.conv2.weight.shape)  # 打印点卷积权重形状

    adapter = WConvAdapter(
        in_features=96,   # 输入通道数
        kernel_size=7,
        padding=3,
        stride=1,
        width=32,      # 中间层通道数
        mlp_ratio=0.5, # 中间层通道数与输入通道数的比例
        groups=4,     # 分组卷积的组数
        den = [0.7, 1.0, 0.7]  # 时间维度的权重设置
    )

    # 测试输入
    x = torch.randn(32, 96, 64, 64)  # [B, C, H, W] 形状的输入
    out = adapter(x)  # 输出形状为 [B, C, H, W]
    print(out.shape)  # 应该输出 torch.Size([32, 96, 64, 64])
