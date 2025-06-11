# Ke Chen
# knutchen@ucsd.edu
# HTS-AT: A HIERARCHICAL TOKEN-SEMANTIC AUDIO TRANSFORMER FOR SOUND CLASSIFICATION AND DETECTION
# Model Core
# below codes are based and referred from https://github.com/microsoft/Swin-Transformer
# Swin Transformer for Computer Vision: https://arxiv.org/pdf/2103.14030.pdf


import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


from models.components.mixture_of_existing_adapters import MixtureOfExistingAdapters
from models.components.model_utilities import (PatchEmbed, Mlp, DropPath,
                                               get_linear_layer, get_conv2d_layer)
from models.components.model_utilities_adapt import ConvAdapterDesign1
from models.components.utils import trunc_normal_, to_2tuple, interpolate

# below codes are based and referred from https://github.com/microsoft/Swin-Transformer
# Swin Transformer for Computer Vision: https://arxiv.org/pdf/2103.14030.pdf

ADAPT_CONFIG = {}

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """基于窗口的多头自注意力(W-MSA)模块，支持相对位置偏置。
    支持移位和非移位窗口。
    参数:
        dim (int): 输入通道数
        window_size (tuple[int]): 窗口的高度和宽度
        num_heads (int): 注意力头的数量
        qkv_bias (bool, optional): 是否在query、key、value中添加可学习的偏置
        qk_scale (float | None, optional): 覆盖默认的qk缩放比例 head_dim ** -0.5
        attn_drop (float, optional): 注意力权重的dropout比率
        proj_drop (float, optional): 输出的dropout比率
        ADAPT_CONFIG: 适配器配置参数
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., ADAPT_CONFIG=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 定义相对位置偏置的参数表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # 获取窗口内每个token的相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # 将索引从0开始
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # 确保ADAPT_CONFIG可用，如果未传入则初始化为空字典
        if ADAPT_CONFIG is None:
            ADAPT_CONFIG = {}

        # 定义QKV投影层
        self.qkv = get_linear_layer(in_features=dim, 
                                  out_features=dim * 3, 
                                  bias=qkv_bias, 
                                  ADAPT_CONFIG=ADAPT_CONFIG)
        self.attn_drop = nn.Dropout(attn_drop)
        
        # 定义输出投影层
        self.proj = get_linear_layer(in_features=dim, 
                                   out_features=dim, 
                                   ADAPT_CONFIG=ADAPT_CONFIG)
        self.proj_drop = nn.Dropout(proj_drop)

        # 初始化相对位置偏置表
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        # 获取适配器配置
        adapt_kwargs_global = ADAPT_CONFIG.get('adapt_kwargs', {})
        adapter_method = ADAPT_CONFIG.get('method', '')
        self.adapter_type = adapt_kwargs_global.get('type', '')
        adapter_position = adapt_kwargs_global.get('position', [])
        
        self.adapter_instance = None  # 统一的适配器实例

        print('==========WindowAttention初始化=============')
        print(f'ADAPT_CONFIG方法: {adapter_method}')
        print(f'全局adapt_kwargs类型: {self.adapter_type}')
        print(f'全局adapt_kwargs位置: {adapter_position}')

        # 根据配置初始化不同类型的适配器
        if 'SpatialAdapter' in adapter_position:
            if self.adapter_type == 'linear_adapter':  # 原始简单适配器
                print("启用的是 Adapter for WindowAttention")
                from models.components.model_utilities_adapt import Adapter
                self.adapter_instance = Adapter(dim, **adapt_kwargs_global)
            elif self.adapter_type == 'adapter_dct':
                print("启用的是 DCTAdapter for WindowAttention")
                from models.components.model_utilities_adapt import DCTAdapter
                self.adapter_instance = DCTAdapter(
                    in_features=dim,
                    **adapt_kwargs_global
                )
            elif self.adapter_type == 'adapter_frequency':
                print("启用的是 DCTFrequencyAdapter for WindowAttention")
                from models.components.model_utilities_adapt import DCTFrequencyAdapter
                self.adapter_instance = DCTFrequencyAdapter(
                    in_features=dim,
                    **adapt_kwargs_global
                )
            elif self.adapter_type == 'adapter_se':
                print("启用的是 SEAdapter for WindowAttention")
                from models.components.model_utilities_adapt import SEAdapter
                self.adapter_instance = SEAdapter(
                    in_features=dim,
                    **adapt_kwargs_global
                )
            elif self.adapter_type == 'conv_adapter':
                print("启用的是 ConvAdapterDesign1 for WindowAttention")
                from models.components.model_utilities_adapt import ConvAdapterDesign1
                self.adapter_instance = ConvAdapterDesign1(
                    in_features=dim,
                    **adapt_kwargs_global
                )
            elif self.adapter_type == 'wConvAdapter':
                print("启用的是 wConvAdapter for WindowAttention")
                from models.components.model_utilities_adapt import WConvAdapter
                self.adapter_instance = WConvAdapter(
                    # inplanes=dim,
                    # outplanes=dim,
                    in_features=dim,
                    **adapt_kwargs_global
                )
            elif self.adapter_type == 'mixture_existing':
                print("启用的是 混合适配器 for WindowAttention")
                from models.components.mixture_of_existing_adapters import MixtureOfExistingAdapters
                # 获取各种配置参数
                experts_config = adapt_kwargs_global.get('experts_config', None)
                dct_expert_config = adapt_kwargs_global.get('dct_expert_kwargs', {})
                freq_expert_config = adapt_kwargs_global.get('freq_expert_kwargs', {})
                adapter_config = adapt_kwargs_global.get('adapter_kwargs',{})
                router_config = adapt_kwargs_global.get('router_kwargs', {})
                gate_noise = adapt_kwargs_global.get('gate_noise_factor', 1.0)
                aux_loss_coeff = adapt_kwargs_global.get('aux_loss_coeff', 0.01)

                # 初始化混合适配器
                self.adapter_instance = MixtureOfExistingAdapters(
                    dim,
                    experts_config=experts_config,
                    dct_adapter_kwargs=dct_expert_config,
                    freq_adapter_kwargs=freq_expert_config,
                    adapter_kwargs=adapter_config,
                    router_kwargs=router_config,
                    gate_noise_factor=gate_noise,
                    aux_loss_coeff=aux_loss_coeff
                )
            else:
                print("没有匹配的适配器类型或 'SpatialAdapter' 不在位置列表中。")
               
        else:
            print("'SpatialAdapter' 不在适配器位置列表中。WindowAttention 将不会启用适配器。")
            # from models.components.model_utilities_adapt import Adapter
            # self.adapter_instance = Adapter(dim, **adapt_kwargs_global)
            # print('已偷偷启动普通Adapter在WindowAttention中')
        
        print('==========WindowAttention初始化完成=============')

    def forward(self, x, mask=None):
        """
        前向传播函数
        参数:
            x: 输入特征，形状为 (num_windows*B, N, C)
            mask: 掩码，形状为 (num_windows, Wh*Ww, Wh*Ww) 或 None
        """
        B_, N, C = x.shape  #(num_windows*B, N, C)
        # 计算QKV
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # 缩放Q
        q = q * self.scale
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1))

        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        # 应用掩码（如果存在）
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # 计算输出
        x_main = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x_main = self.proj(x_main)
        # attn.shape is [B, num_heads, 补丁数量, 补丁数量]
        # 应用适配器（如果存在）       x_main.shape is [B, 补丁的数量, 每个补丁的特征维度]
        adapted_x_main = 0.0
        if self.adapter_instance is not None:

            if isinstance(self.adapter_instance, MixtureOfExistingAdapters):
                adapted_x_main = self.adapter_instance(x_main)
            else:
                if self.adapter_type == 'wConvAdapter':
                    B = x_main.shape[0]
                    N = x_main.shape[1]
                    C = x_main.shape[2]

                    H = W = int(math.sqrt(N))
                    adapted_x_main = self.adapter_instance(x_main.transpose(1, 2).reshape(B, C, H, W))
                    adapted_x_main = adapted_x_main.reshape(B, N, C)
                else:
                    adapted_x_main = self.adapter_instance(x_main)

            x_main = adapted_x_main + x_main
            
        # if aux_loss_from_adapter is not None:
            #     current_aux_loss += aux_loss_from_adapter
        x_main = self.proj_drop(x_main)
        return x_main, attn

    def extra_repr(self):
        """返回模块的额外表示信息"""
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


# We use the model based on Swintransformer Block, therefore we can use the swin-transformer pretrained model
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_before_mlp='ln'):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm_before_mlp = norm_before_mlp
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, ADAPT_CONFIG=ADAPT_CONFIG)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.norm_before_mlp == 'ln':
            self.norm2 = nn.LayerNorm(dim)
        elif self.norm_before_mlp == 'bn':
            self.norm2 = lambda x: nn.BatchNorm1d(dim)(x.transpose(1, 2)).transpose(1, 2)
        else:
            raise NotImplementedError
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop, **ADAPT_CONFIG)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        # 获取适配器配置
        self.adapt_kwargs_global = ADAPT_CONFIG.get('adapt_kwargs', {})
        self.adapter_method = ADAPT_CONFIG.get('method', '')
        self.adapter_type = self.adapt_kwargs_global.get('type', '')
        self.adapter_position = self.adapt_kwargs_global.get('position', [])
        
         # 初始化适配器
        self.adapter = None
        print('======================SwinTransformerBlock=====================')
        if self.adapter_type == 'conv_adapter' and 'before_msa' in self.adapter_position:
            print('启用的是ConvAdapterDesign1在before_msa')
            # 初始化ConvAdapterDesign1
            self.adapter = ConvAdapterDesign1(
                in_features=dim,  # 使用dim作为输入特征维度
                **self.adapt_kwargs_global
            )
        elif self.adapter_type == 'linear_adapter' and 'before_msa' in self.adapter_position:
            print('启用的是LinearAdapter在before_msa')
            # 初始化LinearAdapter
            from models.components.model_utilities_adapt import Adapter
            self.adapter = Adapter(
                in_features=dim,
                **self.adapt_kwargs_global
            )
        print('===================SwinTransformerBlock========================')

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape    # B表示batch_size, L维度, C为通道

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        '''
        
            这里我要加入残差串行的Adapter
            在MSHA和LayerNorm之前加入Adapter
            先进行局部卷积，在进行全局MHSA
            X raw is [B, L, C]

        '''
        x_conv = shortcut.view(B, H, W, C).permute(0, 3, 1, 2)
        x_linear = shortcut
        # 如果配置了使用ConvAdapterDesign1
        if self.adapter_type == 'conv_adapter' and 'before_msa' in self.adapter_position:
            if self.adapter is not None:
                # 应用ConvAdapter
                x = self.adapter(x_conv, residual=shortcut)
                x = x.view(B, H, W, C)
                
        elif self.adapter_type == 'linear_adapter' and 'before_msa' in self.adapter_position: 
            if self.adapter is not None:
                x = self.adapter(x_linear, residual=shortcut)
                x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        x_windows_attended, attn_matrix = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = x_windows_attended.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn_matrix

    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"



class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        # self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.reduction = get_linear_layer(in_features=4 * dim, 
                                          out_features = 2 * dim, 
                                          bias=False, **ADAPT_CONFIG)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self):
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 norm_before_mlp='ln',):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer, norm_before_mlp=norm_before_mlp)
            for i in range(depth)])

        # patch merging layer 仅仅就是一个下采样
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        attns = []
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x, attn = blk(x)
                if not self.training:
                    attns.append(attn.unsqueeze(0))
        if self.downsample is not None:
            x = self.downsample(x)
        if not self.training:
            attn = torch.cat(attns, dim = 0)
            attn = torch.mean(attn, dim = 0)
        return x, attn

    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


# The Core of HTSAT
class HTSAT_Swin_Transformer(nn.Module):
    r"""HTSAT based on the Swin Transformer
    Args:
        spec_size (int | tuple(int)): Input Spectrogram size. Default 256
        patch_size (int | tuple(int)): Patch size. Default: 4
        path_stride (iot | tuple(int)): Patch Stride for Frequency and Time Axis. Default: 4
        in_chans (int): Number of input image channels. Default: 1 (mono)
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each HTSAT-Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 8
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        config (module): The configuration Module from config.py
    """

    def __init__(self, in_chans=7, spec_size=256, patch_size=4, 
                 patch_stride=(4,4), embed_dim=96, depths=[2, 2, 6, 2], 
                 num_heads=[4, 8, 16, 32], window_size=8, mlp_ratio=4., 
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., mel_bins=64,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, 
                 patch_norm=True, norm_before_mlp='ln', cfg_adapt={}):
        super(HTSAT_Swin_Transformer, self).__init__()
        self.spec_size = spec_size 
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.ape = ape
        self.in_chans = in_chans
        self.num_heads = num_heads
        self.num_layers = len(self.depths)
        self.num_features = int(self.embed_dim * 2 ** (self.num_layers - 1))
        self.time_res = patch_stride[1] * 2 ** (self.num_layers - 1)
        
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate

        self.qkv_bias = qkv_bias
        self.qk_scale = None

        self.patch_norm = patch_norm
        self.norm_layer = norm_layer if self.patch_norm else None
        self.norm_before_mlp = norm_before_mlp
        self.mlp_ratio = mlp_ratio
        
        self.mel_bins = mel_bins
        self.freq_ratio = self.spec_size // mel_bins

        # split spectrogram into non-overlapping patches
        global ADAPT_CONFIG
        ADAPT_CONFIG = cfg_adapt
        
        self.patch_embed = PatchEmbed(
            img_size=self.spec_size, patch_size=self.patch_size, in_chans=self.in_chans, 
            embed_dim=self.embed_dim, norm_layer=self.norm_layer, patch_stride = patch_stride, 
            **cfg_adapt)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.grid_size
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        
        self.pos_drop = nn.Dropout(p=self.drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]

        # build layers 这里的num_layers是4 group，depths是[2, 2, 6, 2]，num_heads是[4, 8, 16, 32]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(self.embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate,
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                norm_layer=self.norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                norm_before_mlp=self.norm_before_mlp,)
            self.layers.append(layer)
        
        self.norm = self.norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.SF = self.spec_size // (2 ** (len(self.depths) - 1)) \
            // self.patch_stride[0] // self.freq_ratio

    
    # Reshape the wavform to a img size, if you want to use the pretrained swin transformer model
    def reshape_wav2img(self, x):
        '''
        x: mel spectrogram, (batch_size, in_chans, time_steps, spec_size) 
        '''
        target_T = int(self.spec_size * self.freq_ratio)
        target_F = self.spec_size // self.freq_ratio

        T = x.shape[2]

        # pad the time axis to the target_T
        x = nn.functional.pad(x, (0, 0, 0, target_T - T))

        x = x.permute(0,1,3,2).contiguous() # (B,C,F,T)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], self.freq_ratio, 
                      x.shape[3] // self.freq_ratio) # (B,C,F,r,T//r)
        x = x.permute(0,1,3,2,4).contiguous() # (B,C,r,F,T//r)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3], 
                      x.shape[4]) # (B,C,r*F,T//r)
        return x

    def forward_features(self, x):
        frames_num = x.shape[2]  # 保存输入特征的帧数，用于后续计算
        x = self.patch_embed(x)  # (B, N, C) 将输入特征转换为序列形式的patch嵌入向量 N表示patch数量/序列长度，C表示每个补丁的特征维度
        if self.ape:
            x = x + self.absolute_pos_embed  # 如果启用，添加绝对位置编码
        x = self.pos_drop(x)  # 应用位置编码的dropout
        for i, layer in enumerate(self.layers):
            x, attn = layer(x)  # 依次通过所有Transformer层，每层返回特征和注意力权重
        
        # 对最终特征进行处理
        x = self.norm(x)  # 应用层归一化
        B, N, C = x.shape  # B:批次大小, N:序列长度(patch数量), C:特征维度
        SF = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[0]  # 计算特征的频率维度
        ST = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[1]  # 计算特征的时间维度
        x = x.permute(0,2,1).contiguous().reshape(B, C, SF, ST)  # 重塑为(B,C,F,T)形式
        B, C, F, T = x.shape  # 获取重塑后的维度大小
        
        # 处理频率维度以适合后续任务
        c_freq_bin = F // self.freq_ratio  # 计算频率分箱数
        x = x.reshape(B, C, F // c_freq_bin, c_freq_bin, T)  # 重塑以分离频率维度
        x = x.permute(0,1,3,2,4).contiguous().reshape(B, C, c_freq_bin, -1)  # 调整维度顺序并重塑为最终输出形式

        return x  # 返回处理后的特征表示
        
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch_size, in_chans, spec_size, time_steps)
        """
        '''
            将音频频谱图重新排列为适合视觉Transformer处理的二维图像格式
            调整频率和时间分辨率以匹配预训练模型的输入要求
            这是将音频特征适配为视觉模型输入的关键步骤
        '''

        x = self.reshape_wav2img(x)  # (B,C,r*F,T//r) 
        
        '''
            raw x.shape is [B, C, F, T] [B, 7, 1001, 64]
            after reshape_wav2img, x.shape is [B, C, r*F, T//r] [B, 7, 256, 256]
        '''

        '''
            执行Swin Transformer的主要特征提取过程
            包括分层次的特征提取、自注意力机制和位置信息处理
            生成能够表示音频事件时间、频率和空间特性的高级特征表示
        '''
        x = self.forward_features(x)
        
        return x

    def forward_patch(self, x):
        x = self.reshape_wav2img(x) # (B,C,r*F,T//r)
        x = self.patch_embed(x) # (B, N, C)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        return x
    
    def forward_reshape(self, x):
        frames_num = self.spec_size
        x = self.norm(x)
        B, N, C = x.shape
        SF = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[0]
        ST = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[1]
        x = x.permute(0,2,1).contiguous().reshape(B, C, SF, ST)
        B, C, F, T = x.shape

        c_freq_bin = F // self.freq_ratio
        x = x.reshape(B, C, F // c_freq_bin, c_freq_bin, T)
        x = x.permute(0,1,3,2,4).contiguous().reshape(B, C, c_freq_bin, -1)

        return x
    
    
    

        
