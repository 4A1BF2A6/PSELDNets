from pathlib import Path
import h5py
import numpy as np
from .components.data import BaseDataset
import soundfile as sf

def load_audio(path, index_begin, index_end):
    """加载音频文件
    
    参数:
        path: 音频文件路径
        index_begin: 开始索引
        index_end: 结束索引
        
    返回:
        加载的音频数据，转置后的形状为 [channels, samples]
    """
    try:
        # 尝试直接读取指定区间的音频
        x = sf.read(path, dtype='float32', start=index_begin, stop=index_end)[0].T
    except:
        # 如果读取失败，则读取整个文件后截取
        x = sf.read(path, dtype='float32')[0].T
        x = x[:, index_begin:index_end]
    return x

def generate_spatial_samples(audio, method, **kwargs):
    """生成空间音频样本
    
    注意：仅支持单声源目标
    
    参数:
        audio: 输入音频数据
        method: 生成方法，可选 'einv2', 'accdoa', 'multi_accdoa'
        **kwargs: 其他参数，包括标签数据
        
    返回:
        处理后的音频和标签数据
    """
    # 如果是双通道音频，只取第一个通道
    if audio.ndim == 2:
        audio = audio[0]

    # 随机生成方位角和仰角
    azi = np.random.randint(-180, 180)  # 方位角范围：-180到180度
    ele = np.random.randint(-90, 90)    # 仰角范围：-90到90度
    w = audio
    # 计算空间坐标
    x = np.cos(np.deg2rad(azi)) * np.cos(np.deg2rad(ele))
    y = np.sin(np.deg2rad(azi)) * np.cos(np.deg2rad(ele))
    z = np.sin(np.deg2rad(ele))
    # 生成四通道音频数据 [W, Y, Z, X]
    audio = np.stack((w, y * audio, z * audio, x * audio), axis=0)

    # 根据不同方法处理标签
    if method == 'einv2':
        # EINV2方法：处理SED和DOA标签
        sed_label, doa_label = kwargs['sed_label'], kwargs['doa_label']
        assert sed_label.sum(axis=-2).max() <= 1  # 确保每个时间步最多只有一个声源
        doa_label = np.zeros_like(doa_label)
        # 更新DOA标签
        doa_label[..., 0, 0] = sed_label.sum(axis=(-1, -2)) * x
        doa_label[..., 0, 1] = sed_label.sum(axis=(-1, -2)) * y
        doa_label[..., 0, 2] = sed_label.sum(axis=(-1, -2)) * z
        return audio, sed_label, doa_label
        
    elif method == 'accdoa':
        # ACCDOA方法：处理ACCDOA标签
        accdoa_label = kwargs['accdoa_label']
        num_classes = accdoa_label.shape[-1] // 4
        se_label = accdoa_label[:, :num_classes]
        assert se_label.sum(axis=-1).max() <= 1  # 确保每个时间步最多只有一个声源
        accdoa_label = np.zeros_like(accdoa_label)
        # 更新ACCDOA标签
        accdoa_label[..., num_classes:num_classes*2] = x * se_label
        accdoa_label[..., num_classes*2:num_classes*3] = y * se_label
        accdoa_label[..., num_classes*3:] = z * se_label
        return audio, accdoa_label
        
    elif method == 'multi_accdoa':
        # 多声源ACCDOA方法：处理ADPIT标签
        adpit_label = kwargs['adpit_label']
        num_classes = adpit_label.shape[-1]
        se_label = adpit_label[:, :, 0, :]
        assert se_label.sum(axis=(-1, -2)).max() <= 1  # 确保每个时间步最多只有一个声源
        adpit_label = np.zeros_like(adpit_label)
        # 更新ADPIT标签
        adpit_label[:, :, 0, :] = se_label
        adpit_label[:, :, 1, :] = x * se_label
        adpit_label[:, :, 2, :] = y * se_label
        adpit_label[:, :, 3, :] = z * se_label
        return audio, adpit_label


class DatasetACCDOA(BaseDataset):
    """ACCDOA方法的数据集类"""
    
    def __getitem__(self, idx):
        """获取数据集中的单个样本
        
        参数:
            idx: 样本索引
            
        返回:
            包含音频数据和标签的字典
        """
        # 获取数据段信息
        clip_indexes = self.segments_list[idx]
        path, segments = clip_indexes[0], clip_indexes[1:]
        fn = Path(path).stem
        index_begin = segments[0]
        index_end = segments[1]
        pad_width_before = segments[2]
        pad_width_after = segments[3]
        
        # 加载音频数据
        if self.audio_feature in ['logmelIV']:
            # 在线特征提取
            data_path = path
            x = load_audio(data_path, index_begin, index_end)
            pad_width = ((0, 0), (pad_width_before, pad_width_after))
            dataset = path.split('/')[-3]
        else:
            # 离线特征提取
            data_path = self.data_dir.joinpath(path)
            with h5py.File(data_path, 'r') as hf:
                x = hf['feature'][:, index_begin: index_end] 
            pad_width = ((0, 0), (pad_width_before, pad_width_after), (0, 0))
            dataset = path.split('/')[-2]
        x = np.pad(x, pad_width, mode='constant')

        # 加载标签数据（非测试集）
        if self.dataset_type != 'test':
            meta_path = self.accdoa_label_dir.joinpath('{}.h5'.format(dataset))
            index_begin_label = int(index_begin / self.points_per_predictions)
            index_end_label = int(index_end / self.points_per_predictions)
            with h5py.File(meta_path, 'r') as hf:
                # 读取并处理标签
                se_label = hf[f'{fn}/accdoa/se'][index_begin_label: index_end_label, ...].astype(np.float32)
                azi_label = hf[f'{fn}/accdoa/azi'][index_begin_label: index_end_label, ...].astype(np.float32)
                ele_label = hf[f'{fn}/accdoa/ele'][index_begin_label: index_end_label, ...].astype(np.float32)
                # 计算空间坐标
                lx = np.cos(np.deg2rad(azi_label)) * np.cos(np.deg2rad(ele_label)) * se_label
                ly = np.sin(np.deg2rad(azi_label)) * np.cos(np.deg2rad(ele_label)) * se_label
                lz = np.sin(np.deg2rad(ele_label)) * se_label
                del azi_label, ele_label
                # 合并标签
                accdoa_label = np.concatenate((se_label, lx, ly, lz), axis=1, dtype=np.float32)
                
            # 处理标签填充
            pad_width_after_label = int(
                self.chunklen_sec[self.dataset_type] / self.label_res - accdoa_label.shape[0])
            if pad_width_after_label != 0:
                accdoa_label_new = np.zeros(
                    (pad_width_after_label, 4*self.num_classes), 
                    dtype=np.float32)
                accdoa_label = np.concatenate((accdoa_label, accdoa_label_new), axis=0)
                
        # 训练集数据增强
        if self.dataset_type == 'train' and self.cfg.adapt.method == 'mono_adapter':
            x, accdoa_label = generate_spatial_samples(
                x, method='accdoa', accdoa_label=accdoa_label)
                
        # 构建返回样本
        if self.dataset_type != 'test':
            ov = str(max(np.sum(accdoa_label[:, :self.num_classes], axis=1).max().astype(int), 1))
            sample = {
                'filename': path,
                'data': x,
                'accdoa_label': accdoa_label[:, self.num_classes:],
                'ov': ov
            }
        else:
            sample = {
                'filename': path,
                'data': x
            }
          
        return sample    


class DatasetEINV2(BaseDataset):

    def __getitem__(self, idx):
        """ Datset for the EINV2 method of SELD task

        """
        clip_indexes = self.segments_list[idx]
        path, segments = clip_indexes[0], clip_indexes[1:]
        fn = Path(path).stem
        index_begin = segments[0]
        index_end = segments[1]
        pad_width_before = segments[2]
        pad_width_after = segments[3]
        if self.audio_feature in ['logmelIV']:
            data_path = path
            x = load_audio(data_path, index_begin, index_end)
            pad_width = ((0, 0), (pad_width_before, pad_width_after))
            dataset = path.split('/')[-3]
        else:
            data_path = self.data_dir.joinpath(path)
            with h5py.File(data_path, 'r') as hf:
                x = hf['feature'][:, index_begin: index_end] 
            pad_width = ((0, 0), (pad_width_before, pad_width_after), (0, 0))
            dataset = path.split('/')[-2]
        x = np.pad(x, pad_width, mode='constant')
        if self.dataset_type != 'test':
            meta_path = self.track_label_dir.joinpath('{}.h5'.format(dataset))
            index_begin_label = int(index_begin / self.points_per_predictions)
            index_end_label = int(index_end / self.points_per_predictions)
            with h5py.File(meta_path, 'r') as hf:
                sed_label = hf[f'{fn}/sed_label'][index_begin_label: index_end_label, :self.max_ov]
                doa_label = hf[f'{fn}/doa_label'][index_begin_label: index_end_label, :self.max_ov]
            pad_width_after_label = int(
                self.chunklen_sec[self.dataset_type] / self.label_res - sed_label.shape[0])
            if pad_width_after_label != 0:
                sed_label_new = np.zeros((pad_width_after_label, self.max_ov, self.num_classes))
                sed_label = np.concatenate((sed_label, sed_label_new), axis=0)
                doa_label_new = np.zeros((pad_width_after_label, self.max_ov, 3))
                doa_label = np.concatenate((doa_label, doa_label_new), axis=0)
        if self.dataset_type == 'train' and self.cfg.adapt.method == 'mono_adapter':
            x, sed_label, doa_label = generate_spatial_samples(
                x, method='einv2', sed_label=sed_label, doa_label=doa_label)
        if self.dataset_type != 'test':
            ov = str(max(np.sum(sed_label, axis=(1,2)).max().astype(int), 1))
            sample = {
                'filename': path,
                'data': x,
                'sed_label': sed_label.astype(np.float32),
                'doa_label': doa_label.astype(np.float32),
                'ov': ov
            }
        else:
            sample = {
                'filename': path,
                'data': x
            }
          
        return sample    


class DatasetMultiACCDOA(BaseDataset):
    """多声源ACCDOA方法的数据集类
    
    该类继承自BaseDataset，用于处理多声源场景下的声音事件定位和检测任务。
    使用ADPIT（Activity-Dependent Parameterized Independent Tracking）标签格式。
    """

    def __getitem__(self, idx):
        """获取数据集中的单个样本
        
        参数:
            idx: 样本索引
            
        返回:
            dict: 包含音频数据和标签的字典
                - filename: 音频文件路径
                - data: 音频特征数据
                - adpit_label: ADPIT格式的标签数据（非测试集）
                - ov: 重叠声源数量（非测试集）
        """
        # 获取数据段信息
        clip_indexes = self.segments_list[idx]
        path, segments = clip_indexes[0], clip_indexes[1:]  # 分离文件路径和段信息
        fn = Path(path).stem  # 获取文件名（不含扩展名）
        index_begin = segments[0]  # 开始索引
        index_end = segments[1]    # 结束索引
        pad_width_before = segments[2]  # 前填充宽度
        pad_width_after = segments[3]   # 后填充宽度

        # 加载音频特征数据
        if self.audio_feature in ['logmelIV']:
            # 在线特征提取：直接加载音频文件
            data_path = path
            x = load_audio(data_path, index_begin, index_end)
            pad_width = ((0, 0), (pad_width_before, pad_width_after))
            dataset = path.split('/')[-3]
        else:
            # 离线特征提取：从预计算的特征文件中加载
            data_path = self.data_dir.joinpath(path)
            with h5py.File(data_path, 'r') as hf:
                x = hf['feature'][:, index_begin: index_end] 
            pad_width = ((0, 0), (pad_width_before, pad_width_after), (0, 0))
            dataset = path.split('/')[-2]
        # 对特征数据进行填充
        x = np.pad(x, pad_width, mode='constant')

        # 非测试集：加载标签数据
        if 'test' not in self.dataset_type:
            # 构建标签文件路径
            meta_path = self.adpit_label_dir.joinpath('{}.h5'.format(dataset))
            # 计算标签的起始和结束索引
            index_begin_label = int(index_begin / self.points_per_predictions)
            index_end_label = int(index_end / self.points_per_predictions)
            
            # 从HDF5文件中加载标签数据
            with h5py.File(meta_path, 'r') as hf:
                # 加载声源活动检测(SED)标签
                se_label = hf[f'{fn}/adpit/se'][index_begin_label: index_end_label, ...].astype(np.float32)
                # 加载方位角标签
                azi_label = hf[f'{fn}/adpit/azi'][index_begin_label: index_end_label, ...].astype(np.float32)
                # 加载仰角标签
                ele_label = hf[f'{fn}/adpit/ele'][index_begin_label: index_end_label, ...].astype(np.float32)
                
                # 计算空间坐标
                lx = np.cos(np.deg2rad(azi_label)) * np.cos(np.deg2rad(ele_label)) * se_label
                ly = np.sin(np.deg2rad(azi_label)) * np.cos(np.deg2rad(ele_label)) * se_label
                lz = np.sin(np.deg2rad(ele_label)) * se_label
                
                # 释放不需要的变量
                del azi_label, ele_label
                
                # 堆叠标签数据：[时间步, 声源数, 特征维度, 类别数]
                adpit_label = np.stack((se_label, lx, ly, lz), axis=2)
                
            # 计算标签填充
            pad_width_after_label = int(
                self.chunklen_sec[self.dataset_type] / self.label_res - adpit_label.shape[0])
            if pad_width_after_label != 0:
                # 创建填充标签
                adpit_label_new = np.zeros((pad_width_after_label, 6, 4, self.num_classes), dtype=np.float32)
                # 连接原始标签和填充标签
                adpit_label = np.concatenate((adpit_label, adpit_label_new), axis=0)

        # 训练集数据增强：生成空间音频样本
        if self.dataset_type == 'train' and self.cfg.adapt.method == 'mono_adapter':
            x, adpit_label = generate_spatial_samples(
                x, method='multi_accdoa', adpit_label=adpit_label)

        # 构建返回样本
        if 'test' not in self.dataset_type:
            # 计算重叠声源数量
            ov = str(max(np.sum(adpit_label[:, :, 0, :], axis=(1,2)).max().astype(int), 1))
            
            # 非测试集：返回完整样本
            sample = {
                'filename': path,
                'data': x,
                'adpit_label': adpit_label,
                'ov': ov
            }
        else:
            # 测试集：只返回文件名和特征数据
            sample = {
                'filename': path,
                'data': x
            }
          
        return sample    