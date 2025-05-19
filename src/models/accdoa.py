import torch
import torch.nn as nn

from models.components.conformer import ConformerBlocks
from models.components.model_utilities import Decoder, get_conv2d_layer
from models.components.backbone import CNN8, CNN12
from models.components.htsat import HTSAT_Swin_Transformer
from models.components.passt import PaSST
from models.components.utils import interpolate


class CRNN(nn.Module):
    def __init__(self, cfg, num_classes, in_channels=7, encoder='CNN8', pretrained_path=None,
                 audioset_pretrain=True, num_features=[32, 64, 128, 256]):
        super().__init__()

        data = cfg.data
        mel_bins = cfg.data.n_mels
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.label_res = 0.1
        self.interpolate_time_ratio = 2 ** 3
        self.output_frames = None #int(data.train_chunklen_sec / 0.1)
        self.pred_res = int(data.sample_rate / data.hoplen * self.label_res) # 10
        
        self.scalar = nn.ModuleList([nn.BatchNorm2d(mel_bins) for _ in range(in_channels)])
        if encoder == 'CNN8':
            self.convs = CNN8(in_channels, num_features)
        elif encoder == 'CNN12':
            self.convs = CNN12(in_channels, num_features)
            if pretrained_path:
                print('Loading pretrained model from {}...'.format(pretrained_path))
                self.load_ckpts(pretrained_path, audioset_pretrain)
        else:
            raise NotImplementedError(f'encoder {encoder} is not implemented')
        
        self.num_features = num_features

        self.decoder = Decoder(cfg.model.decoder, num_features[-1], 
                               num_layers=cfg.model.num_decoder_layers)
        self.fc = nn.Linear(num_features[-1], 3*num_classes, bias=True, )
        self.final_act = nn.Tanh()
    
    def load_ckpts(self, pretrained_path, audioset_pretrain=True):
        if audioset_pretrain:
            CNN14_ckpt = torch.load(pretrained_path, map_location='cpu')['model']
            CNN14_ckpt['conv_block1.conv1.weight'] = nn.Parameter(
                CNN14_ckpt['conv_block1.conv1.weight'].repeat(1, self.in_channels, 1, 1) / self.in_channels)
            missing_keys, unexpected_keys = self.convs.load_state_dict(CNN14_ckpt, strict=False)
            assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
            for ich in range(self.in_channels):
                self.scalar[ich].weight.data.copy_(CNN14_ckpt['bn0.weight'])
                self.scalar[ich].bias.data.copy_(CNN14_ckpt['bn0.bias'])
                self.scalar[ich].running_mean.copy_(CNN14_ckpt['bn0.running_mean'])
                self.scalar[ich].running_var.copy_(CNN14_ckpt['bn0.running_var'])
                self.scalar[ich].num_batches_tracked.copy_(CNN14_ckpt['bn0.num_batches_tracked'])
        else:
            ckpt = torch.load(pretrained_path, map_location='cpu')['state_dict']
            ckpt = {k.replace('net.', ''): v for k, v in ckpt.items()}
            ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()} # if compiling the model
            for key, value in self.state_dict().items():
                if key.startswith('fc.'): print(f'Skipping {key}...')
                else: value.data.copy_(ckpt[key])

    def forward(self, x):
        """
        x: waveform, (batch_size, num_channels, time_frames, mel_bins)
        """

        N, _, T, _ = x.shape
        self.output_frames = int(T // self.pred_res)

        # Compute scalar
        x = x.transpose(1, 3)
        for nch in range(x.shape[-1]):
            x[..., [nch]] = self.scalar[nch](x[..., [nch]])
        x = x.transpose(1, 3)

        # encoder
        x = self.convs(x)
        x = x.mean(dim=3) # (N, C, T)
        
        # decoder
        x = x.permute(0, 2, 1) # (N, T, C)
        x = self.decoder(x) # (N, T, C)
        
        x = interpolate(x, ratio=self.interpolate_time_ratio) # (N, T, C)
        x = x.reshape(N, self.output_frames, self.pred_res, -1).mean(dim=2) # (N, T, C)

        # fc
        x = self.final_act(self.fc(x))
        
        return {
            'accdoa': x,
        }


class ConvConformer(CRNN):
    def __init__(self, cfg, num_classes, in_channels=7, encoder='CNN8', pretrained_path=None, 
                 audioset_pretrain=True, num_features=[32, 64, 128, 256]):
        super().__init__(cfg, num_classes, in_channels, encoder, pretrained_path, 
                         audioset_pretrain, num_features)

        self.decoder = ConformerBlocks(encoder_dim=self.num_features[-1], num_layers=2)
    

class HTSAT(nn.Module):
    def __init__(self, cfg, num_classes, in_channels=7, audioset_pretrain=True,
                 pretrained_path='ckpts/HTSAT-fullset-imagenet-768d-32000hz.ckpt', 
                 **kwargs):
        super().__init__()
        
        data = cfg.data
        mel_bins = cfg.data.n_mels
        cfg_adapt = cfg.adapt
        self.label_res = 0.1
        self.num_classes = num_classes
        self.output_frames = None # int(data.train_chunklen_sec / 0.1)
        self.tgt_output_frames = int(10 / 0.1) # 10-second clip input to the model
        self.pred_res = int(data.sample_rate / data.hoplen * self.label_res)
        self.in_channels = in_channels
        
        # scalar - 标量归一化层
        # 为每个输入通道创建一个BatchNorm2d层，用于对梅尔频谱图进行归一化处理
        # mel_bins: 梅尔频谱图的频率维度大小
        # in_channels: 输入通道数，通常是7（对应7个麦克风通道）
        self.scalar = nn.ModuleList([nn.BatchNorm2d(mel_bins) for _ in range(in_channels)])
        
        # encoder - 编码器
        # 使用HTSAT-Swin Transformer作为特征提取器
        # in_channels: 输入通道数
        # mel_bins: 梅尔频谱图的频率维度大小
        # cfg_adapt: 适配器配置参数
        # **kwargs: 其他可选参数
        self.encoder = HTSAT_Swin_Transformer(in_channels, mel_bins=mel_bins, 
                                              cfg_adapt=cfg_adapt, **kwargs)
        
        # tscam_conv - 时间-空间卷积注意力模块
        # 将编码器提取的特征转换为ACCDOA（活动声音事件的方向和距离）表示
        # in_channels: 输入通道数，等于编码器的特征维度
        # out_channels: 输出通道数 = 类别数 * 3（每个类别对应xyz三个坐标）
        # kernel_size: 卷积核大小
        #   - 频率维度使用全部频带(self.encoder.SF)
        #   - 时间维度为3，用于捕获时间上下文信息
        # padding: 在时间维度上填充1，保持时间维度不变
        self.tscam_conv = nn.Conv2d(
            in_channels = self.encoder.num_features,
            out_channels = self.num_classes * 3,
            kernel_size = (self.encoder.SF,3),
            padding = (0,1))

        # fc - 全连接层
        # 使用恒等映射，不做额外变换
        # 在子类中可能会被重写为其他变换
        self.fc = nn.Identity()

        # final_act - 最终激活函数
        # 使用Tanh激活函数，将输出值限制在[-1,1]范围内
        # 这符合ACCDOA表示中方向向量的范围要求
        self.final_act = nn.Tanh()

        if pretrained_path:
            print('Loading pretrained model from {}...'.format(pretrained_path))
            self.load_ckpts(pretrained_path, audioset_pretrain)
        
        self.freeze_layers_if_needed(cfg_adapt.get('method', ''))
        # 解冻tscam_conv层，因为tscam_conv层是用于适应不同输入通道的，所以需要解冻
        self.tscam_conv.requires_grad_(True)

    def freeze_layers_if_needed(self, adapt_method):
        # 如果适配方法名中不包含"adapter"，则不进行任何操作直接返回
        if 'adapter' not in adapt_method:
            return
        
        # 初始化ADAPTER标志为False，用于跟踪是否找到适配器层
        ADAPTER = False
        
        # 冻结整个模型的所有参数（设置requires_grad=False）
        self.requires_grad_(False)
        print('\n Freezing the model...')
        
        ''' 适配器微调处理：适用于单声道/多通道音频剪辑 '''
        # 遍历模型中的所有参数
        for name, param in self.named_parameters():
            # 对所有偏置参数(bias)保持可训练状态
            if 'bias' in name: 
                param.requires_grad_(True)
                print(f'param {name} is trainable')
            
            # 对所有适配器(adapter)或LoRA参数保持可训练状态
            if 'adapter' in name or 'lora' in name:
                # 标记找到了适配器层
                ADAPTER = True
                param.requires_grad_(True)
                print(f'param {name} is trainable')
        
        ''' 使用单声道剪辑进行微调的特殊处理 '''
        # 如果方法是mono_adapter且没有找到适配器层，则进行完全微调
        if adapt_method == 'mono_adapter' and not ADAPTER:
            print('No adapter found, all parameters are trainable')
            # 设置所有参数为可训练状态(完全微调)
            self.requires_grad_(True)

    def load_ckpts(self, pretrained_path, audioset_pretrain=True):
        if audioset_pretrain:
            print('AudioSet-pretrained model...')
            htsat_ckpts = torch.load(pretrained_path, map_location='cpu')['state_dict']
            htsat_ckpts = {k.replace('sed_model.', ''): v for k, v in htsat_ckpts.items()}
            for key, value in self.encoder.state_dict().items():
                try:
                    if key == 'patch_embed.proj.weight':
                        paras = htsat_ckpts[key].repeat(1, self.in_channels, 1, 1) / self.in_channels
                        value.data.copy_(paras)
                    elif 'tscam_conv' not in key and 'head' not in key and 'adapter' not in key:
                        value.data.copy_(htsat_ckpts[key])
                    else: print(f'Skipping {key}...')
                except: print(key, value.shape, htsat_ckpts[key].shape)
            for ich in range(self.in_channels):
                self.scalar[ich].weight.data.copy_(htsat_ckpts['bn0.weight'])
                self.scalar[ich].bias.data.copy_(htsat_ckpts['bn0.bias'])
                self.scalar[ich].running_mean.copy_(htsat_ckpts['bn0.running_mean'])
                self.scalar[ich].running_var.copy_(htsat_ckpts['bn0.running_var'])
                self.scalar[ich].num_batches_tracked.copy_(htsat_ckpts['bn0.num_batches_tracked'])
        else:
            print('DataSynthSELD-pretrained model...')
            ckpt = torch.load(pretrained_path, map_location='cpu')['state_dict']
            ckpt = {k.replace('net.', ''): v for k, v in ckpt.items()}
            ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()} # if compiling the model
            for idx, (key, value) in enumerate(self.state_dict().items()):
                if key.startswith(('fc.', 'head.', 'tscam_conv.')) or 'lora' in key or 'adapter' in key:
                    print(f'{idx+1}/{len(self.state_dict())}: Skipping {key}...')
                else:
                    try: value.data.copy_(ckpt[key])
                    except: print(f'{idx+1}/{len(self.state_dict())}: {key} not in ckpt.dict, skipping...')

    def forward(self, x):
        """
        x: waveform, (batch_size, num_channels, time_frames, mel_bins)
        """

        B, C, T, F = x.shape

        # Concatenate clips to a 10-second clip if necessary
        if self.output_frames is None:
            self.output_frames = int(T // self.pred_res)
        if self.output_frames < self.tgt_output_frames:
            assert self.output_frames == self.tgt_output_frames // 2, \
                'only support 5-second or 10-second clip or input to the model'
            factor = 2
            assert B % factor == 0, 'batch size should be a factor of {}'.format(factor)
            x = torch.cat((x[:B//factor, :, :-1], x[B//factor:, :, :-1]), dim=2)
        elif self.output_frames > self.tgt_output_frames:
            raise NotImplementedError('output_frames > tgt_output_frames is not implemented')

        # Compute scalar
        x = x.transpose(1, 3)
        for nch in range(x.shape[-1]):
            x[..., [nch]] = self.scalar[nch](x[..., [nch]])
        x = x.transpose(1, 3)

        x = self.encoder(x)
        x = self.tscam_conv(x)
        x = torch.flatten(x, 2) # (B, C, T) B是批次大小 C是通道数 T是时间步长
        x = x.permute(0,2,1).contiguous() # B, T, C
        x = self.fc(x)

        # Match the output shape
        x = interpolate(x, ratio=self.encoder.time_res, method='bilinear')
        x = x[:, :self.output_frames * self.pred_res]
        if self.output_frames < self.tgt_output_frames:
            x = torch.cat((x[:, :self.output_frames], x[:, self.output_frames:]), dim=0)
        x = x.reshape(B, self.output_frames, self.pred_res, -1).mean(dim=2)

        x = self.final_act(x)

        return {
            'accdoa': x,
        }


class PASST(nn.Module):
    def __init__(self, cfg, num_classes, in_channels=7, pretrained_path=None,
                 audioset_pretrain=True, **kwargs):
        super().__init__()
        
        mel_bins = cfg.data.n_mels
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # scalar
        self.scalar = nn.ModuleList([nn.BatchNorm2d(mel_bins) for _ in range(in_channels)])
        # encoder
        self.encoder = PaSST(in_channels, **kwargs)
        # fc
        self.fc = nn.Linear(self.encoder.num_features, num_classes * 3)
        self.final_act = nn.Tanh()

        if pretrained_path:
            print('Loading pretrained model from {}...'.format(pretrained_path))
            self.load_ckpts(pretrained_path, audioset_pretrain)

    def load_ckpts(self, pretrained_path, audioset_pretrain=True):
        if audioset_pretrain:
            passt_ckpt = torch.load(pretrained_path, map_location='cpu')
            for key, value in self.encoder.state_dict().items():
                if key == 'patch_embed.proj.weight':
                    paras = passt_ckpt[key].repeat(1, self.in_channels, 1, 1) / self.in_channels
                    value.data.copy_(paras)
                elif key == 'time_new_pos_embed':
                    time_new_pos_embed = passt_ckpt[key]
                    ori_time_len = time_new_pos_embed.shape[-1]
                    targ_time_len = self.encoder.time_new_pos_embed.shape[-1]
                    if ori_time_len >= targ_time_len:
                        start_index = int((ori_time_len - targ_time_len) / 2)
                        self.encoder.time_new_pos_embed.data.copy_(
                            time_new_pos_embed[:, :, :, start_index:start_index+targ_time_len])
                    else:
                        self.encoder.time_new_pos_embed.data.copy_(nn.functional.interpolate(
                            time_new_pos_embed, size=(1, targ_time_len), mode='bilinear'))
                elif key == 'freq_new_pos_embed':
                    freq_new_pos_embed = passt_ckpt[key]
                    ori_freq_len = freq_new_pos_embed.shape[-2]
                    targ_freq_len = self.encoder.freq_new_pos_embed.shape[-2]
                    if ori_freq_len >= targ_freq_len:
                        start_index = int((ori_freq_len - targ_freq_len) / 2)
                        self.encoder.freq_new_pos_embed.data.copy_(
                            freq_new_pos_embed[:, :, start_index:start_index+targ_freq_len, :])
                    else:
                        self.encoder.freq_new_pos_embed.data.copy_(nn.functional.interpolate(
                            freq_new_pos_embed, size=(1, targ_freq_len), mode='bilinear'))
                elif 'head' in key: 
                    if key in ['head.0.weight', 'head.0.bias']:
                        value.data.copy_(passt_ckpt[key])
                else:
                    value.data.copy_(passt_ckpt[key])
        else:
            ckpt = torch.load(pretrained_path, map_location='cpu')['state_dict']
            ckpt = {k.replace('net.', ''): v for k, v in ckpt.items()}
            ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()} # if compiling the model
            for key, value in self.state_dict().items():
                if key.startswith('fc.'): print(f'Skipping {key}...')
                else: value.data.copy_(ckpt[key])
    
    def forward(self, x):
        """
        x: waveform, (batch_size, num_channels, time_frames, mel_bins)
        """

        # Compute scalar
        x = x.transpose(1, 3)
        for nch in range(x.shape[-1]):
            x[..., [nch]] = self.scalar[nch](x[..., [nch]])
        x = x.transpose(1, 3)

        x = self.encoder(x)[0]
        x = self.fc(x)
        x = self.final_act(x)

        return {
            'accdoa': x,
        }


        
    
