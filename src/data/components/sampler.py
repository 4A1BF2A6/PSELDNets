import numpy as np
from torch.utils.data import Sampler
import torch.distributed as dist

class UserDistributedBatchSampler(Sampler):
    """用户自定义的分布式批处理采样器，仅用于训练集。
    
    这个采样器支持多GPU分布式训练，确保每个GPU获得不同的数据批次。
    """
    def __init__(self, clip_num, batch_size=1, seed=2023, data_indices=None,
                 shuffle=True, last_batch_supplement=True):
        # 获取当前进程的rank和总进程数
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        self.clip_num = clip_num
        # 总批次大小 = 每个GPU的批次大小 * GPU数量
        self.batch_size = batch_size * self.num_replicas

        # 初始化数据索引
        if data_indices is None:
            self.indices = np.arange(clip_num)
        else:
            self.indices = data_indices
            self.clip_num = len(data_indices)
        
        self.pointer = 0  # 当前处理位置指针
        self.shuffle = shuffle  # 是否打乱数据
        if self.shuffle:
            # 使用固定的随机种子确保可重复性
            self.random_state = np.random.RandomState(seed)
            self.random_state.shuffle(self.indices)
        
        # 补充最后一个不完整的批次
        if last_batch_supplement:
            padding_size = self.batch_size - self.clip_num % self.batch_size
            self.indices = np.append(self.indices, self.indices[:padding_size])
            self.clip_num = self.clip_num + padding_size
    
    def __iter__(self):
        """
        返回: 
            batch_indices (int): 批次数据的索引
        """   
        while True:
            # 当处理完所有数据时，重置指针并可能重新打乱数据
            if self.pointer >= self.clip_num:
                self.pointer = 0
                if self.shuffle:
                    self.random_state.shuffle(self.indices)

            # 为每个GPU分配不同的数据批次
            # 使用步长为GPU数量的切片确保每个GPU获得不同的数据
            batch_indices = self.indices[self.pointer+self.rank: self.pointer+self.batch_size: self.num_replicas]
            self.pointer += self.batch_size
            yield batch_indices

    def __len__(self):
        # 返回总批次数
        return np.ceil(self.clip_num / self.batch_size).astype(int)


class UserBatchSampler(Sampler):
    """用户自定义的批处理采样器，仅用于训练集。
    
    这是单GPU版本的采样器，不支持分布式训练。
    """
    def __init__(self, clip_num, batch_size=1, seed=2023, data_indices=None,
                 shuffle=True, last_batch_supplement=True):
        self.clip_num = clip_num
        self.batch_size = batch_size

        # 初始化数据索引
        if data_indices is None:
            self.indices = np.arange(clip_num)
        else:
            self.indices = data_indices
            self.clip_num = len(data_indices)
        
        self.pointer = 0  # 当前处理位置指针
        self.shuffle = shuffle  # 是否打乱数据
        if self.shuffle:
            # 使用固定的随机种子确保可重复性
            self.random_state = np.random.RandomState(seed)
            self.random_state.shuffle(self.indices)
        
        # 补充最后一个不完整的批次
        if last_batch_supplement:
            padding_size = self.batch_size - self.clip_num % self.batch_size
            self.indices = np.append(self.indices, self.indices[:padding_size])
            self.clip_num = self.clip_num + padding_size
    
    def __iter__(self):
        """
        返回: 
            batch_indices (int): 批次数据的索引
        """   
        while True:
            # 当处理完所有数据时，重置指针并可能重新打乱数据
            if self.pointer >= self.clip_num:
                self.pointer = 0
                if self.shuffle:
                    self.random_state.shuffle(self.indices)

            # 获取当前批次的数据索引
            batch_indices = self.indices[self.pointer: self.pointer+self.batch_size]
            self.pointer += self.batch_size
            yield batch_indices

    def __len__(self):
        # 返回总批次数
        return np.ceil(self.clip_num / self.batch_size).astype(int)
