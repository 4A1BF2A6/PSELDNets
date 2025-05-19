import numpy as np
import lightning as L
import torch
from torch.utils.data import DataLoader, ConcatDataset
from .components.sampler import UserDistributedBatchSampler
from utils.utilities import get_pylogger
from collections import OrderedDict

# 获取日志记录器
log = get_pylogger(__name__)

class SELDDataModule(L.LightningDataModule):
    """声音事件定位和检测(SELD)任务的数据模块
    
    继承自PyTorch Lightning的LightningDataModule，用于管理数据加载和预处理
    """

    # 导入数据集类
    from data.data import DatasetEINV2, DatasetACCDOA, DatasetMultiACCDOA
    
    # 定义可用的数据集类型字典
    UserDataset = {
        'accdoa': DatasetACCDOA,      # ACCDOA方法的数据集
        'einv2': DatasetEINV2,        # EINV2方法的数据集
        'multi_accdoa': DatasetMultiACCDOA,  # 多声源ACCDOA方法的数据集
    }

    def __init__(self, cfg, dataset, stage='fit'):
        """初始化数据模块
        
        参数:
            cfg: 配置对象
            dataset: 数据集对象
            stage: 阶段，可选 'fit'（训练）或 'test'（测试）
        """
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset        
        self.seed = cfg.seed  # 随机种子

        # 初始化路径字典和验证集标签格式字典
        self.paths_dict = OrderedDict()
        self.valid_gt_dcaseformat = OrderedDict()

        # 获取模型方法
        method = cfg.model.method

        if stage == 'fit':
            # 训练阶段初始化
            self.train_set, self.val_set = [], []

            # 加载训练数据集
            for dataset_name, rooms in cfg.data.train_dataset.items():
                self.train_set.append(
                    self.UserDataset[method](cfg, dataset, dataset_name, rooms, 'train'))
            self.train_set = ConcatDataset(self.train_set)  # 合并训练数据集

            # 加载验证数据集
            for dataset_name, rooms in cfg.data.valid_dataset.items():
                self.val_set.append(
                    self.UserDataset[method](cfg, dataset, dataset_name, rooms, 'valid'))
                # 更新路径字典和验证集标签格式
                self.paths_dict.update(self.val_set[-1].paths_dict)
                self.valid_gt_dcaseformat.update(self.val_set[-1].valid_gt_dcaseformat)
            # TODO: 临时实现，需要支持多数据加载器
            self.val_set = ConcatDataset(self.val_set)  # 合并验证数据集

            # 设置训练批次大小
            self.train_batch_size = cfg['model']['batch_size']
            log.info(f"训练片段数量: {len(self.train_set)}")
            log.info(f"验证片段数量: {len(self.val_set)}")

        elif stage == 'test':
            # 测试阶段初始化
            self.test_set = []
            # 加载测试数据集
            for dataset_name, rooms in cfg.data.test_dataset.items():
                self.test_set.append(
                    self.UserDataset[method](cfg, dataset, dataset_name, rooms, 'test'))
                self.paths_dict.update(self.test_set[-1].paths_dict)
            self.test_batch_size = cfg['model']['batch_size']
            self.test_set = ConcatDataset(self.test_set)  # 合并测试数据集
            log.info(f"测试片段数量: {len(self.test_set)}")
    
    def train_dataloader(self):
        """创建训练数据加载器
        
        返回:
            DataLoader: 训练数据加载器，使用自定义的分布式批次采样器
        """
        # 创建分布式批次采样器
        batch_sampler = UserDistributedBatchSampler(
            clip_num=len(self.train_set), 
            batch_size=self.train_batch_size,
            seed=self.seed)
        log.info(f"每个epoch的批次数: {len(batch_sampler)}")
        
        # 创建数据加载器
        return DataLoader(
            dataset=self.train_set,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.num_workers,  # 数据加载的工作进程数
            generator=torch.Generator().manual_seed(self.seed),  # 设置随机种子
            pin_memory=True)  # 使用固定内存
    
    def val_dataloader(self):
        """创建验证数据加载器
        
        返回:
            DataLoader: 验证数据加载器
        """
        rank = self.trainer.local_rank
        world_size = self.trainer.world_size
        # 每个GPU使用不同的采样器（与DistributedSampler相同）
        # self.val_set.segments_list = self.val_set.segments_list[rank::world_size]

        return DataLoader(
            dataset=self.val_set,
            batch_size=self.train_batch_size,
            shuffle=False,  # 验证集不打乱
            num_workers=self.cfg.num_workers,
            generator=torch.Generator().manual_seed(self.seed),
            pin_memory=True)
    
    def test_dataloader(self):
        """创建测试数据加载器
        
        返回:
            DataLoader: 测试数据加载器
        """
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.test_batch_size,
            shuffle=False,  # 测试集不打乱
            num_workers=self.cfg.num_workers,
            generator=torch.Generator().manual_seed(self.seed),
            pin_memory=True)