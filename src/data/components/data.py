from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils.data_utilities import load_output_format_file

# 支持wav格式的数据集列表
wav_format_datasets = ['official', 'STARSS23', 'DCASE2021', 'L3DAS22', 'synth', 'official2024', 'STARSS22']

class BaseDataset(Dataset):
    """SELD任务的基础数据集类
    
    该类实现了PyTorch的Dataset接口，用于处理声音事件定位和检测(SELD)任务的数据集
    """
    def __init__(self, cfg, dataset, dataset_name, rooms, dataset_type='train'):
        """初始化数据集
        
        参数:
            cfg: 配置对象，包含数据处理的各项参数
            dataset: 数据集对象，包含数据集的基本信息
            dataset_name: 数据集名称
            rooms: 房间列表，指定要使用的房间数据
            dataset_type: 数据集类型，可选 'train'|'valid'|'test'
                - 'train'和'valid'仅用于训练阶段
                - 'valid'或'test'用于推理阶段
        """
        super().__init__()

        # 保存基本配置
        self.cfg = cfg
        self.dataset_type = dataset_type
        self.label_res = dataset.label_resolution  # 标签分辨率
        self.max_ov = dataset.max_ov              # 最大重叠数
        self.num_classes = dataset.num_classes    # 类别数量

        # 从配置中读取音频处理参数
        self.sample_rate = cfg['data']['sample_rate']        # 采样率
        self.audio_feature = cfg['data']['audio_feature']    # 音频特征类型
        self.audio_type = cfg['data']['audio_type']         # 音频类型
        # 设置不同阶段的数据块长度（秒）
        self.chunklen_sec = {
            'train': cfg['data']['train_chunklen_sec'],
            'valid': cfg['data']['test_chunklen_sec'],
            'test': cfg['data']['test_chunklen_sec'],}
        # 设置不同阶段的数据块步长（秒）
        self.hoplen_sec = {
            'train': cfg['data']['train_hoplen_sec'], 
            'valid': cfg['data']['test_hoplen_sec'],
            'test': cfg['data']['test_hoplen_sec'],}

        # 设置数据目录
        hdf5_dir = Path(cfg.paths.hdf5_dir)
        # 确定数据集阶段：如果是mix或split5则为eval，否则为dev
        dataset_stage = 'eval' if rooms == ['mix'] or rooms == ['split5'] else 'dev'
        
        # 根据特征类型设置数据目录和预测点数
        if self.audio_feature in ['logmelIV', 'logmel']:
            # 对于logmelIV或logmel特征，支持在线提取
            main_data_dir = hdf5_dir.joinpath(f'data/{self.sample_rate}fs/wav')
            self.points_per_predictions = self.sample_rate * self.label_res
        else:
            # 其他特征类型需要离线提取
            main_data_dir = hdf5_dir.joinpath(f'data/{self.sample_rate}fs/feature')
            self.data_dir = main_data_dir.joinpath(dataset_stage, self.audio_feature)
            self.points_per_predictions = int(
                self.label_res / (cfg['data']['hoplen'] / self.sample_rate))
        
        # 设置标签目录
        label_dir = hdf5_dir.joinpath('label')
        self.track_label_dir = label_dir.joinpath('track/{}'.format(dataset_stage))    # 轨迹标签
        self.accdoa_label_dir = label_dir.joinpath('accdoa/{}'.format(dataset_stage))  # ACCDOA标签
        self.adpit_label_dir = label_dir.joinpath('adpit/{}'.format(dataset_stage))    # ADPIT标签
        
        # 处理房间列表
        if not (rooms == ['mix'] or rooms == ['split5']): 
            rooms = [room+'_' for room in rooms]
        rooms.sort()
        
        # 根据数据集类型选择索引文件
        if self.dataset_type == 'train':
            indexes_path = main_data_dir.joinpath('{}/{}_{}sChunklen_{}sHoplen_train.csv'.format(
                dataset_stage, dataset_name, self.chunklen_sec['train'], self.hoplen_sec['train']))
        elif self.dataset_type in ['valid', 'test']:
            indexes_path = main_data_dir.joinpath('{}/{}_{}sChunklen_{}sHoplen_test.csv'.format(
                dataset_stage, dataset_name, self.chunklen_sec['test'], self.hoplen_sec['test']))
        print(indexes_path)
        
        # 读取数据段索引
        segments_indexes = pd.read_csv(indexes_path, header=None).values
        # 根据房间列表筛选数据段
        self.segments_list = [_segment for _segment in segments_indexes 
                            for _room in rooms if _room in _segment[0]]
        print(f'{dataset_name} {dataset_type} dataset: {len(self.segments_list)} segments')
        
        # 对于非wav格式数据集，将文件扩展名从.wav改为.flac
        if dataset_name not in wav_format_datasets:
            for i in range(len(self.segments_list)):
                self.segments_list[i][0] = self.segments_list[i][0].replace('.wav', '.flac')

        # 对于验证和测试集，创建路径字典
        if self.dataset_type in ['valid', 'test']:
            self.paths_dict = OrderedDict() # {path: num_frames}
            for segment in self.segments_list:
                self.paths_dict[segment[0]] = int(
                    np.ceil(segment[2] / self.points_per_predictions))
                    
        # 对于验证集，加载元数据
        if self.dataset_type == 'valid':
            self.valid_gt_dcaseformat = OrderedDict() # {path: metrics_dict}
            for segment in self.segments_list:
                if segment[0] not in self.valid_gt_dcaseformat:
                    # 构建元数据文件路径
                    metafile = segment[0].replace('foa', 'metadata').replace('.flac', '.csv')
                    if dataset_name in wav_format_datasets: 
                        metafile = metafile.replace('.wav', '.csv')
                    if dataset_name == 'L3DAS22':
                        metafile = metafile.replace('/data_', '/metadata_')
                    # 加载元数据文件
                    self.valid_gt_dcaseformat[segment[0]] = load_output_format_file(metafile)

    def __len__(self):
        """返回数据集的长度"""
        return len(self.segments_list)

    def __getitem__(self, idx):
        """获取数据集中的单个样本
        
        参数:
            idx: 样本索引
            
        返回:
            数据样本（需要子类实现具体逻辑）
        """
        raise NotImplementedError


