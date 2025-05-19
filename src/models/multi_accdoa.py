import torch.nn as nn
from models import accdoa
from utils.utilities import get_pylogger

log = get_pylogger(__name__)

class CRNN(accdoa.CRNN):
    def __init__(self, *args, **kwargs):
        super(CRNN, self).__init__(*args, **kwargs)
        self.fc = nn.Linear(self.num_features[-1], 
                            3 * 3 * self.num_classes, bias=True)

    def forward(self, x):
        return {
            'multi_accdoa': super().forward(x)['accdoa']
        }

class ConvConformer(accdoa.ConvConformer):
    def __init__(self, *args, **kwargs):
        super(ConvConformer, self).__init__(*args, **kwargs)
        self.fc = nn.Linear(self.num_features[-1], 
                            3 * 3 * self.num_classes, bias=True)

    def forward(self, x):
        return {
            'multi_accdoa': super().forward(x)['accdoa']
        }

class HTSAT(accdoa.HTSAT):
    """多轨道ACCDOA实现的HTSAT模型
    
    这个类扩展了基本的HTSAT模型，添加了多轨道支持，可以同时检测多个重叠的声音事件
    """
    def __init__(self, *args, **kwargs):
        """初始化多轨道HTSAT模型
        
        参数继承自基类accdoa.HTSAT
        """
        super(HTSAT, self).__init__(*args, **kwargs)
        self.tscam_conv = nn.Conv2d(
            in_channels = self.encoder.num_features,    # 使用编码器的特征维度作为输入通道
            out_channels = self.num_classes * 3 * 3,    # 输出通道数 = 类别数 * 3(xyz坐标) * 3(轨道数)
                                                        # 支持最多3个同时发生的声音事件
            kernel_size = (self.encoder.SF, 3),         # 卷积核大小：频率维度使用全部频带，时间维度为3
            padding = (0, 1))                           # 在时间维度上进行填充
        self.fc = nn.Identity()                         # 使用恒等映射，不做额外变换
        log.info(f'Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}')  # 记录可训练参数数量
        log.info(f'Non-trainable parameters: {sum(p.numel() for p in self.parameters() if not p.requires_grad)}')  # 记录冻结参数数量

    def forward(self, x):
        """前向传播函数
        
        Args:
            x: 输入特征，通常是梅尔频谱图
            
        Returns:
            包含'multi_accdoa'键的字典，值为多轨道ACCDOA表示
            每个轨道可以表示一个声音事件的类别和方向
        """
        return {
            'multi_accdoa': super().forward(x)['accdoa']  # 调用父类的forward方法获取accdoa输出，并重命名为multi_accdoa
        }

class PASST(accdoa.PASST):
    def __init__(self, *args, **kwargs):
        super(PASST, self).__init__(*args, **kwargs)
        self.fc = nn.Linear(self.encoder.num_features, 
                            3 * 3 * self.num_classes)

    def forward(self, x):
        return {
            'multi_accdoa': super().forward(x)['accdoa']
        }