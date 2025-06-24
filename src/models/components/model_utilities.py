import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils import to_2tuple
from models.components.conformer import ConformerBlocks
from .model_utilities_adapt import Adapter
from .mixture_of_existing_adapters import MixtureOfExistingAdapters

def get_linear_layer(method='', rir_simulate='', *args, **kwargs):
    kwargs.pop('ADAPT_CONFIG', None)  # 移除 ADAPT_CONFIG，避免传递给 nn.Linear
    # method = method.split('_')
    adapt_kws = kwargs.pop('adapt_kwargs', {}) if 'adapt_kwargs' in kwargs else {}
    if method == 'lora':
        from models.components.model_utilities_adapt import Linear as LinearLoRA
        kwargs.update(kwargs.get('linear_kwargs', {}))
        kwargs = {k: v for k, v in kwargs.items() if '_kwargs' not in k}

        return LinearLoRA(*args, **kwargs)
    else:
        kwargs = {k: v for k, v in kwargs.items() if '_kwargs' not in k}

        return nn.Linear(*args, **kwargs)

def get_conv2d_layer(method='', rir_simulate='', **kwargs):
    # method = method.split('_')
    if 'lora' in method:
        from .model_utilities_adapt import Conv2d as Conv2dLoRA
        kwargs.update(kwargs.get('conv_kwargs', {}))
        kwargs = {k: v for k, v in kwargs.items() if '_kwargs' not in k}
        return Conv2dLoRA(**kwargs)
    else:
        kwargs = {k: v for k, v in kwargs.items() if '_kwargs' not in k}
        return nn.Conv2d(**kwargs)


class CrossStitch(nn.Module):
    def __init__(self, feat_dim):

        super().__init__()
        self.weight = nn.Parameter(
            torch.FloatTensor(feat_dim, 2, 2).uniform_(0.1, 0.9)
            )
    
    def forward(self, x, y):
        if x.dim() == 4:
            equation = 'c, nctf -> nctf'
        elif x.dim() == 3:
            equation = 'c, ntc -> ntc'
        else:
            raise ValueError('x must be 3D or 4D tensor')
        x = torch.einsum(equation, self.weight[:, 0, 0], x) + \
            torch.einsum(equation, self.weight[:, 0, 1], y)
        y = torch.einsum(equation, self.weight[:, 1, 0], x) + \
            torch.einsum(equation, self.weight[:, 1, 1], y)
        return x, y


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, 
                kernel_size=(3,3), stride=(1,1), padding=(1,1),
                dilation=1, bias=False,
                pool_size=(2,2), pool_type='avg'):
        super().__init__()

        if pool_type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=pool_size)
        elif pool_type == 'max':
            self.pool = nn.MaxPool2d(kernel_size=pool_size)
        else:
            raise Exception('pool_type must be avg or max')

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                    out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, 
                    out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            self.pool,
        )
        
    def forward(self, x):
        x = self.double_conv(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                kernel_size=(3,3), stride=(1,1), padding=(1,1),
                dilation=1, bias=False,
                pool_size=(2,2), pool_type='avg'):
        
        super(ConvBlock, self).__init__()

        if pool_type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=pool_size)
        elif pool_type == 'max':
            self.pool = nn.MaxPool2d(kernel_size=pool_size)
        else:
            raise Exception('pool_type must be avg or max')
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, bias=bias)
                            
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, bias=bias)
                            
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.,
                 **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        ADAPT_CONFIG = kwargs # kwargs 本身就是从上层传递过来的适配器配置
        adapt_kwargs_global = ADAPT_CONFIG.get('adapt_kwargs', {})

        self.fc1 = get_linear_layer(in_features=in_features, 
                                    out_features=hidden_features,
                                    ADAPT_CONFIG=ADAPT_CONFIG) # 传递ADAPT_CONFIG
        self.act = act_layer()
        self.fc2 = get_linear_layer(in_features=hidden_features, 
                                    out_features=out_features,
                                    ADAPT_CONFIG=ADAPT_CONFIG) # 传递ADAPT_CONFIG
        self.drop = nn.Dropout(drop)

        # 适配器类型判断
        self.current_adapter_type = adapt_kwargs_global.get('type', '')
        self.adapter_position = adapt_kwargs_global.get('position', [])
        self.adapter_instance = None

        is_mlp_adapter_pos = 'MlpAdapter' in self.adapter_position
        
        print(f'============== MLP (in_features={in_features}) =================')
        print(f'ADAPT_CONFIG method: {ADAPT_CONFIG.get("method", "N/A")}')
        print(f'Global adapt_kwargs type: {self.current_adapter_type}')
        print(f'Global adapt_kwargs position: {self.adapter_position}')

        if is_mlp_adapter_pos:
            if self.current_adapter_type == 'linear_adapter':
                print('启用的是Adapter for MLP')
                from .model_utilities_adapt import Adapter
                self.adapter_instance = Adapter(in_features, **adapt_kwargs_global)
            elif self.current_adapter_type == 'adapter_dct':
                print('启用的是DCT适配器 for MLP')
                from .model_utilities_adapt import DCTAdapter
                self.adapter_instance = DCTAdapter(in_features, **adapt_kwargs_global)
            elif self.current_adapter_type == 'adapter_frequency':
                print('启用的是DCTFrequency适配器 for MLP')
                from .model_utilities_adapt import DCTFrequencyAdapter
                self.adapter_instance = DCTFrequencyAdapter(in_features, **adapt_kwargs_global)
            elif self.current_adapter_type == 'adapter_se':
                print('启用的是SE适配器 for MLP')
                from .model_utilities_adapt import SEAdapter
                self.adapter_instance = SEAdapter(in_features, **adapt_kwargs_global)
            elif self.current_adapter_type == 'conv_adapter':
                print('启用的是ConvAdapterDesign1适配器 for MLP')
                from .model_utilities_adapt import ConvAdapterDesign1
                self.adapter_instance = ConvAdapterDesign1(in_features, **adapt_kwargs_global)
            elif self.current_adapter_type == 'wConvAdapter':
                print('启用的是wConvAdapter适配器 for MLP')
                from .model_utilities_adapt import WConvAdapter
                self.adapter_instance = WConvAdapter(
                    # inplanes=in_features,
                    # outplanes=in_features,
                    in_features,
                    **adapt_kwargs_global
                )
            elif self.current_adapter_type == 'mixture_existing': # 混合适配器
                print('启用的是 混合适配器 for MLP')

                # mixture_specific_kwargs 应从 adapt_kwargs_global 中提取
                # 提取专家配置
                experts_config = adapt_kwargs_global.get('experts_config', None)
                # dct_expert_config = adapt_kwargs_global.get('dct_expert_kwargs', {})
                # freq_expert_config = adapt_kwargs_global.get('freq_expert_kwargs', {})
                # adapter_config = adapt_kwargs_global.get('adapter_kwargs', {})
                router_config = adapt_kwargs_global.get('router_kwargs', {})
                gate_noise = adapt_kwargs_global.get('gate_noise_factor', 1.0)
                aux_loss_coeff = adapt_kwargs_global.get('aux_loss_coeff', 0.01)

                self.adapter_instance = MixtureOfExistingAdapters(
                    in_features,
                    experts_config=experts_config,
                    router_kwargs=router_config,
                    gate_noise_factor=gate_noise,
                    aux_loss_coeff=aux_loss_coeff
                )
            elif self.current_adapter_type == 'adapter_mona':
                from .model_utilities_adapt import MonaAdapter
                print('启用的是MonaAdapter适配器 for MLP')
                self.adapter_instance = MonaAdapter(in_features, **adapt_kwargs_global)
            else:
                print('MLP中没有启用任何特定类型的适配器或类型未知')
        else:
            print('MLPAdapter不在当前适配器位置列表中')
            # from models.components.model_utilities_adapt import Adapter
            # self.adapter_instance = Adapter(in_features, **adapt_kwargs_global)
            # print('已偷偷启动普通Adapter在MLP中')
        print('=============  MLP ================')


    def forward(self, x):
        
        adapted_output = 0
        # 主干网络的前向传播
        main_path = self.fc1(x)
        main_path = self.act(main_path)
        main_path = self.drop(main_path)
        main_path = self.fc2(main_path)

        if self.adapter_instance is not None:
            if isinstance(self.adapter_instance, MixtureOfExistingAdapters):
                adapted_output = self.adapter_instance(x) # 适配器作用于 x
            else:
                if self.current_adapter_type == 'wConvAdapter':
                    B = x.shape[0]
                    N = x.shape[1]
                    C = x.shape[2]
                    # 在 Swin Transformer 中，H 和 W 通常是相等的，所以我们可以计算平方根
                    H = W = int(math.sqrt(x.shape[1]))  # 计算 H 和 W
                    # 重塑张量 [B, N, C] -> [B, C, H, W]
                    x = x.transpose(1, 2).reshape(B, C, H, W)
                    adapted_output = self.adapter_instance(x) # 适配器作用于 x
                    adapted_output = adapted_output.reshape(B, N, C)
                else:
                    # 方案1：并联
                    adapted_output = self.adapter_instance(x)
                    # 方案2：我的串联方式
                    # adapted_output = self.adapter_instance(main_path)
                    # 方案3：先残差，再并联（上层代码实现) mona的串联方式

        main_path = main_path + adapted_output # 更新 main_path
        main_path = self.drop(main_path) # 在原始的 HTSAT Swin MLP 中，最后的drop在适配器之后   
         
        # return main_path # 原来的返回值
        return main_path


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, 
                 norm_layer=None, flatten=True, patch_stride=16, padding=True, **kwargs):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patch_stride = to_2tuple(patch_stride)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.grid_size = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        if padding:
            padding = ((patch_size[0] - patch_stride[0]) // 2, 
                       (patch_size[1] - patch_stride[1]) // 2)
        else:
            padding = 0

        self.proj = get_conv2d_layer(in_channels=in_chans, out_channels=embed_dim, 
                                     kernel_size=patch_size, stride=patch_stride, 
                                     padding=padding, **kwargs)
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
        #                       stride=patch_stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Decoder(nn.Module):
    def __init__(self, decoder, num_feats, num_layers=2, **kwargs):
        super().__init__()
        self.num_feats = num_feats
        if decoder == 'gru':
            self.decoder = nn.GRU(input_size=num_feats, hidden_size=num_feats//2, 
                                  num_layers=num_layers, bidirectional=True, 
                                  batch_first=True, **kwargs)
        elif decoder == 'conformer':
            self.decoder = ConformerBlocks(encoder_dim=num_feats, num_layers=num_layers, **kwargs)
        elif decoder == 'transformer':
            self.decoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=num_feats, nhead=8, 
                                           batch_first=True, **kwargs),
                num_layers=num_layers)
        elif decoder is None:
            self.decoder = nn.Identity()
        else:
            raise NotImplementedError(f"{decoder} is not implemented")

    def forward(self, x):
        if isinstance(self.decoder, nn.RNNBase):
            x = self.decoder(x)[0]
        else: x = self.decoder(x)
        return x
