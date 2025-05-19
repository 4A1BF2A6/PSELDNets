import shutil
from pathlib import Path
from timeit import default_timer as timer
import soundfile as sf

import h5py
import numpy as np
import pandas as pd
from utils.feature import (Features_Extractor_MIC, 
                           LogmelIV_Extractor)
from tqdm import tqdm

from utils.data_utilities import (segment_index, load_output_format_file,
                                  convert_output_format_polar_to_cartesian)


class Preprocess:
    
    """预处理音频数据
    
    主要功能：
    1. 提取wav文件并存储为hdf5文件
    2. 提取元数据文件并存储为hdf5文件
    """
    
    def __init__(self, cfg, dataset):
        """初始化预处理类
        
        参数:
            cfg: 配置信息
            dataset: 数据集类
        """
        # 基本配置参数
        self.cfg = cfg
        self.dataset = dataset
        
        # 音频处理参数
        self.fs = cfg.data.sample_rate  # 采样率
        self.hoplen = cfg.data.hoplen   # 帧移
        self.nfft = cfg.data.nfft       # FFT点数
        self.n_mels = cfg.data.n_mels   # Mel滤波器组数量
        self.audio_feature = cfg.data.audio_feature  # 音频特征类型
        self.label_res = dataset.label_resolution    # 标签分辨率
        self.num_classes = dataset.num_classes       # 类别数量
        self.wav_format = cfg.wav_format            # wav文件格式

        # 数据分块参数
        self.train_chunklen_sec = cfg.data.train_chunklen_sec  # 训练数据块长度(秒)
        self.train_hoplen_sec = cfg.data.train_hoplen_sec      # 训练数据块步长(秒)
        self.test_chunklen_sec = cfg.data.test_chunklen_sec    # 测试数据块长度(秒)
        self.test_hoplen_sec = cfg.data.test_hoplen_sec        # 测试数据块步长(秒)
            
        # Path for dataset
        hdf5_dir = Path(cfg.paths.hdf5_dir)

        # Path for extraction of wav
        self.data_dir = {
            'foa': dataset.dataset_dir[cfg.dataset_type]['foa'],
            'mic': dataset.dataset_dir[cfg.dataset_type]['mic']}
        data_dir = hdf5_dir.joinpath('data/{}fs'.format(self.fs))

        self.wav_h5_dir = data_dir.joinpath(f'wav/{cfg.dataset_type}')
        self.data_statistics_path = self.wav_h5_dir.joinpath(f'{cfg.dataset_type}/{cfg.dataset}_statistics.txt')

        # Path for extraction of label
        label_dir = hdf5_dir.joinpath('label')
        self.meta_dir = dataset.dataset_dir[cfg.dataset_type]['meta']
        self.meta_track_path = label_dir.joinpath('track/{}/{}.h5'.format(cfg.dataset_type, cfg.dataset))
        self.meta_accdoa_path = label_dir.joinpath('accdoa/{}/{}.h5'.format(cfg.dataset_type, cfg.dataset))
        self.meta_adpit_path = label_dir.joinpath('adpit/{}/{}.h5'.format(cfg.dataset_type, cfg.dataset))

        # Path for extraction of features
        self.feature_h5_dir = data_dir.joinpath('feature/{}/{}/{}'.format(
            cfg.dataset_type, self.audio_feature, cfg.dataset))

        # Path for indexes of data
        self.data_type = 'wav' if self.audio_feature in ['logmelIV', 'logmel'] else 'feature'
        self.indexes_path_list = [ 
            data_dir.joinpath(self.data_type).joinpath('{}/{}_{}sChunklen_{}sHoplen_train.csv'.format(
                cfg.dataset_type, cfg.dataset, self.train_chunklen_sec, self.train_hoplen_sec)),
            data_dir.joinpath(self.data_type).joinpath('{}/{}_{}sChunklen_{}sHoplen_test.csv'.format(
                cfg.dataset_type, cfg.dataset, self.test_chunklen_sec, self.test_hoplen_sec))]


    def extract_track_label(self):
        """提取用于排列不变训练的轨迹标签，存储为h5文件
        
        主要功能：
        1. 处理多音源场景
        2. 为每个音源分配唯一的轨迹编号
        3. 生成SED(声音事件检测)和DOA(到达方向)标签
        """
        max_polyphony = 6 if self.cfg.dataset == 'STARSS23' else 3
        num_classes = self.num_classes

        self.meta_track_path.parent.mkdir(parents=True, exist_ok=True)
        if self.meta_track_path.is_file():
            self.meta_track_path.unlink()
        hf = h5py.File(self.meta_track_path, 'w')

        meta_list = [path for path in sorted(self.meta_dir.glob('*.csv')) if not path.name.startswith('.')]
        iterator = tqdm(enumerate(meta_list), total=len(meta_list), unit='clips', desc='Extracting Track-wise label')
        for idx, meta_file in iterator:
            fn = meta_file.stem
            df = pd.read_csv(meta_file, header=None, sep=',')
            df = df.values
            num_frames = df[-1, 0] + 1
            sed_label = np.zeros((num_frames, max_polyphony, num_classes), dtype=np.bool_)
            doa_label = np.zeros((num_frames, max_polyphony, 3))
            event_indexes = np.array([[None] * max_polyphony] * num_frames)  # event indexes of all frames
            track_numbers = np.array([[None] * max_polyphony] * num_frames)   # track number of all frames
            for row in df:
                frame_idx = row[0]
                event_idx = row[1]
                track_number = row[2]                
                azi = row[3]
                elev = row[4]
                
                ##### track indexing #####
                # default assign current_track_idx to the first available track
                current_event_indexes = event_indexes[frame_idx]
                current_track_indexes = np.where(current_event_indexes == None)[0].tolist()
                # if current_track_indexes:
                #     continue
                try: 
                    current_track_idx = current_track_indexes[0]    
                except: 
                    print(meta_file, frame_idx, event_idx, track_number, current_event_indexes, current_track_indexes)
                    continue

                # label encode
                azi_rad, elev_rad = azi * np.pi / 180, elev * np.pi / 180
                sed_label[frame_idx, current_track_idx, event_idx] = 1.0
                doa_label[frame_idx, current_track_idx, :] = np.cos(elev_rad) * np.cos(azi_rad), \
                    np.cos(elev_rad) * np.sin(azi_rad), np.sin(elev_rad)
                event_indexes[frame_idx, current_track_idx] = event_idx
                track_numbers[frame_idx, current_track_idx] = track_number

            hf.create_dataset(name=f'{fn}/sed_label', data=sed_label, dtype=np.bool_)
            hf.create_dataset(name=f'{fn}/doa_label', data=doa_label, dtype=np.float32)
            
            tqdm.write('{}, {}'.format(idx, fn))
        
        hf.close()


    def extract_accdoa_label(self):
        """提取用于ACCDOA(活动声音分类和到达方向)的标签
        
        主要功能：
        1. 生成声音事件检测标签
        2. 生成方位角标签
        3. 生成仰角标签
        """
        num_classes = self.num_classes

        def _get_accdoa_labels(meta_dcase_format, num_frames):
            se_label = np.zeros((num_frames, num_classes), dtype=np.bool_)
            azi_label = np.zeros((num_frames, num_classes), dtype=np.int16)
            ele_label = np.zeros((num_frames, num_classes), dtype=np.int8)

            for frame_ind, active_event_list in meta_dcase_format.items():
                if frame_ind < num_frames:
                    for active_event in active_event_list:
                        se_label[frame_ind, active_event[0]] = 1
                        azi_label[frame_ind, active_event[0]] = active_event[1]
                        ele_label[frame_ind, active_event[0]] = active_event[2]

            return se_label, azi_label, ele_label
        
        self.meta_accdoa_path.parent.mkdir(parents=True, exist_ok=True)
        if self.meta_accdoa_path.is_file():
            self.meta_accdoa_path.unlink()
        hf = h5py.File(self.meta_accdoa_path, 'w')

        meta_list = [path for path in sorted(self.meta_dir.glob('*.csv')) if not path.name.startswith('.')]
        iterator = tqdm(enumerate(meta_list), total=len(meta_list), unit='clips', desc='Extracting ACCDOA label')
        for idx, meta_file in iterator:
            fn = meta_file.stem
            df = pd.read_csv(meta_file, header=None, sep=',').values
            num_frames = df[-1, 0] + 1
            meta_dcase_format = load_output_format_file(meta_file)
            se_label, azi_label, ele_label = _get_accdoa_labels(meta_dcase_format, num_frames)
            hf.create_dataset(name=f'{fn}/accdoa/se', data=se_label, dtype=np.bool_)
            hf.create_dataset(name=f'{fn}/accdoa/azi', data=azi_label, dtype=np.int16)
            hf.create_dataset(name=f'{fn}/accdoa/ele', data=ele_label, dtype=np.int8)
            tqdm.write('{}, {}'.format(idx, fn))  

        hf.close()


    def extract_adpit_label(self):
        """提取用于ADPIT(辅助复制排列不变训练)的标签
        
        主要功能：
        1. 处理多音源重叠情况
        2. 为每个音源生成独立的标签
        3. 支持最多6个同时发声的音源
        """

        def _get_adpit_labels_for_file(_desc_file):
            """
            Reads description file and returns classification based SED labels and regression based DOA labels
            for multi-ACCDOA with Auxiliary Duplicating Permutation Invariant Training (ADPIT)

            :param _desc_file: dcase format of the meta file
            :return: label_mat: of dimension [nb_frames, 6, 4(=act+XYZ), max_classes]
            """

            _nb_label_frames = list(_desc_file.keys())[-1]
            _nb_lasses = self.num_classes
            se_label = np.zeros((_nb_label_frames, 6, _nb_lasses))  # [nb_frames, 6, max_classes]
            x_label = np.zeros((_nb_label_frames, 6, _nb_lasses))
            y_label = np.zeros((_nb_label_frames, 6, _nb_lasses))
            z_label = np.zeros((_nb_label_frames, 6, _nb_lasses))

            for frame_ind, active_event_list in _desc_file.items():
                if frame_ind < _nb_label_frames:
                    active_event_list.sort(key=lambda x: x[0])  # sort for ov from the same class
                    active_event_list_per_class = []
                    for i, active_event in enumerate(active_event_list):
                        active_event_list_per_class.append(active_event)
                        if i == len(active_event_list) - 1:  # if the last
                            if len(active_event_list_per_class) == 1:  # if no ov from the same class
                                # a0----
                                active_event_a0 = active_event_list_per_class[0]
                                se_label[frame_ind, 0, active_event_a0[0]] = 1
                                x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[1]
                                y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                                z_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                            elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                                # --b0--
                                active_event_b0 = active_event_list_per_class[0]
                                se_label[frame_ind, 1, active_event_b0[0]] = 1
                                x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[1]
                                y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                                z_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                                # --b1--
                                active_event_b1 = active_event_list_per_class[1]
                                se_label[frame_ind, 2, active_event_b1[0]] = 1
                                x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[1]
                                y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                                z_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                            else:  # if ov with more than 2 sources from the same class
                                # ----c0
                                active_event_c0 = active_event_list_per_class[0]
                                se_label[frame_ind, 3, active_event_c0[0]] = 1
                                x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[1]
                                y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                                z_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                                # ----c1
                                active_event_c1 = active_event_list_per_class[1]
                                se_label[frame_ind, 4, active_event_c1[0]] = 1
                                x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[1]
                                y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                                z_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                                # ----c2
                                active_event_c2 = active_event_list_per_class[2]
                                se_label[frame_ind, 5, active_event_c2[0]] = 1
                                x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[1]
                                y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                                z_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]

                        elif active_event[0] != active_event_list[i + 1][0]:  # if the next is not the same class
                            if len(active_event_list_per_class) == 1:  # if no ov from the same class
                                # a0----
                                active_event_a0 = active_event_list_per_class[0]
                                se_label[frame_ind, 0, active_event_a0[0]] = 1
                                x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[1]
                                y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                                z_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                            elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                                # --b0--
                                active_event_b0 = active_event_list_per_class[0]
                                se_label[frame_ind, 1, active_event_b0[0]] = 1
                                x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[1]
                                y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                                z_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                                # --b1--
                                active_event_b1 = active_event_list_per_class[1]
                                se_label[frame_ind, 2, active_event_b1[0]] = 1
                                x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[1]
                                y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                                z_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                            else:  # if ov with more than 2 sources from the same class
                                # ----c0
                                active_event_c0 = active_event_list_per_class[0]
                                se_label[frame_ind, 3, active_event_c0[0]] = 1
                                x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[1]
                                y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                                z_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                                # ----c1
                                active_event_c1 = active_event_list_per_class[1]
                                se_label[frame_ind, 4, active_event_c1[0]] = 1
                                x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[1]
                                y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                                z_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                                # ----c2
                                active_event_c2 = active_event_list_per_class[2]
                                se_label[frame_ind, 5, active_event_c2[0]] = 1
                                x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[1]
                                y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                                z_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                            active_event_list_per_class = []

            label_mat = np.stack((se_label, x_label, y_label, z_label), axis=2)  # [nb_frames, 6, 4(=act+XYZ), max_classes]
            return label_mat
        
        self.meta_adpit_path.parent.mkdir(parents=True, exist_ok=True)
        if self.meta_adpit_path.is_file():
            self.meta_adpit_path.unlink()
        hf = h5py.File(self.meta_adpit_path, 'w')

        meta_list = [path for path in sorted(self.meta_dir.glob('*.csv')) if not path.name.startswith('.')]
        iterator = tqdm(enumerate(meta_list), total=len(meta_list), unit='it')
        for idx, meta_file in iterator:
            fn = meta_file.stem
            meta_dcase_format = load_output_format_file(meta_file)
            meta_dcase_format = convert_output_format_polar_to_cartesian(meta_dcase_format)
            meta_adpit = _get_adpit_labels_for_file(meta_dcase_format)
            hf.create_dataset(name=f'{fn}/adpit', data=meta_adpit, dtype=np.float32)
            tqdm.write('{}, {}'.format(idx, fn))
        hf.close()


    def extract_index(self):
        """提取数据索引并存储为csv文件
        
        主要功能：
        1. 将音频数据分割成固定长度的块
        2. 生成训练和测试数据的索引
        3. 支持特征数据和原始wav数据
        """
        chunklen_sec = [self.train_chunklen_sec, self.test_chunklen_sec]
        hoplen_sec = [self.train_hoplen_sec, self.test_hoplen_sec]
        last_frame_always_padding = [False, True]

        for idx, indexes_path in enumerate(self.indexes_path_list):
            indexes_path.parent.mkdir(parents=True, exist_ok=True)
            if indexes_path.is_file():
                indexes_path.unlink()
            audio_cnt = 0
            f = open(indexes_path, 'w')
            if self.data_type == 'feature':
                frames_per_prediction = int(self.label_res / (self.hoplen / self.fs))
                paths_list_absolute = [path for path in sorted(self.feature_h5_dir.glob('*.h5')) 
                                       if not path.name.startswith('.')]
                paths_list_relative = [path.relative_to(path.parent.parent) for path in paths_list_absolute]
                chunklen = int(chunklen_sec[idx] / self.label_res * frames_per_prediction) 
                hoplen = int(hoplen_sec[idx] / self.label_res * frames_per_prediction) 
                iterator = tqdm(paths_list_absolute, 
                                total=len(paths_list_absolute), 
                                unit='clips', 
                                desc='Extracting indexes')
                for path in iterator:
                    fn = paths_list_relative[audio_cnt]
                    with h5py.File(path, 'r') as hf:
                        num_frames = hf['feature'][:].shape[1]
                    data = np.zeros((1, num_frames))
                    segmented_indexes, segmented_pad_width = segment_index(
                        data, chunklen, hoplen, 
                        last_frame_always_paddding=last_frame_always_padding[idx])
                    for segmented_pairs in list(zip(segmented_indexes, segmented_pad_width)):
                        f.write('{},{},{},{},{}\n'.format(
                            fn, segmented_pairs[0][0], segmented_pairs[0][1],
                                segmented_pairs[1][0], segmented_pairs[1][1]))
                    audio_cnt += 1
                    tqdm.write('Extract indices for audio feature: {}, {}'.format(audio_cnt, fn))
            elif self.data_type == 'wav':
                chunklen = int(chunklen_sec[idx] * self.fs)     
                hoplen = int(hoplen_sec[idx] * self.fs)
                paths_list = sorted(self.data_dir['foa'].glob('*'+self.wav_format))
                if self.cfg.dataset == 'L3DAS22':
                    paths_list = [path for path in paths_list if '_B.wav' not in str(path)]
                iterator = tqdm(paths_list, total=len(paths_list), unit='it')
                for path in iterator:
                    fn = path.name
                    # data_length = sf.read(path, dtype='float32')[0].shape[0]
                    data_length = sf.info(path).frames
                    data = np.zeros((1, data_length))
                    segmented_indexes, segmented_pad_width = segment_index(
                        data, chunklen, hoplen, 
                        last_frame_always_paddding=last_frame_always_padding[idx])
                    for segmented_pairs in list(zip(segmented_indexes, segmented_pad_width)):
                        f.write('{},{},{},{},{}\n'.format(
                            path, segmented_pairs[0][0], segmented_pairs[0][1],
                                segmented_pairs[1][0], segmented_pairs[1][1]))
                    audio_cnt += 1
                    tqdm.write('Extract indices for wav file: {}, {}'.format(audio_cnt, fn))
            f.close()


    def extract_mic_features(self):
        """从麦克风信号中提取特征并存储为hdf5文件
        
        主要功能：
        1. 提取频谱特征
        2. 支持logmelgcc和salsalite两种特征类型
        3. 保存为h5格式
        """
        print('Extracting {} features starts......\n'.format(self.audio_feature))
        if self.feature_h5_dir.is_dir():
            flag = input(f"HDF5 folder {self.feature_h5_dir} is already existed, delete it? (y/n)").lower()
            if flag == 'y':
                shutil.rmtree(self.feature_h5_dir)
            elif flag == 'n':
                print(f"User select not to remove the HDF5 folder {self.feature_h5_dir}. The process will quit.\n")
                return
        self.feature_h5_dir.mkdir(parents=True)
        af_extractor_mic = Features_Extractor_MIC(self.cfg)
        paths_list = [path for data_dir in self.data_dir['mic'] for path in sorted(data_dir.glob('*'+self.wav_format))]
        iterator = tqdm(enumerate(paths_list), 
                        total=len(paths_list), 
                        unit='clips', 
                        desc='Extracting features')
        for count, file in iterator:
            fn = file.stem
            feature_path =self.feature_h5_dir.joinpath(fn+'.h5')
            waveform = sf.read(file, dtype='float32')[0]
            nb_feat_frams = int(len(waveform) / self.hoplen)
            spect = af_extractor_mic._spectrogram(waveform, nb_feat_frams)
            # spect: [n_frames, n_freqs, n_chs]
            if self.audio_feature == 'logmelgcc':
                logmel_spec = af_extractor_mic._get_logmel_spectrogram(spect)
                # logmel_spec: [n_frames, n_mels, n_chs]
                gcc = af_extractor_mic._get_gcc(spect)
                # gcc: [n_frames, n_mels, n_chs]
                feature = np.concatenate((logmel_spec, gcc), axis=-1).transpose((2,0,1))
                # feature: [n_chs, n_frames, n_mels]
                print('feature shape: ', feature.shape)
            elif self.audio_feature == 'salsalite':
                feature = af_extractor_mic._get_salsalite(spect)
            with h5py.File(feature_path, 'w') as hf:
                hf.create_dataset('feature', data=feature, dtype=np.float32)
            tqdm.write('{}, {}, features: {}'.format(count, fn, feature.shape))
        iterator.close()
        print('Extracting {} features finished!'.format(self.audio_feature))


    def extract_l3das22_label(self):
        """提取L3DAS22数据集的标签用于评估
        
        主要功能：
        1. 将原始标签转换为元数据格式
        2. 量化时间戳到帧分辨率
        3. 计算方位角和仰角
        """

        num_frames = int(self.dataset.clip_length / self.label_res)
        print('Converting l3dass22 label files to metadata files starts......\n')

        label_dir = self.dataset.dataset_dir[self.cfg.dataset_type]['label']
        if self.meta_dir.is_dir():
            shutil.rmtree(self.meta_dir)
        self.meta_dir.mkdir(parents=True)
        #quantize time stamp to step resolution
        quantize = lambda x: round(float(x) / self.label_res)
        
        label_list = sorted(label_dir.glob('*.csv'))
        iterator = tqdm(label_list, unit='it')
        for idy, path in enumerate(iterator): # label path
            frame_label = {}
            for i in range(num_frames):
                frame_label[i] = []
            df = pd.read_csv(path)
            fn = path.stem
            fn = fn.replace('label_split', 'split')
            meta_path = self.meta_dir.joinpath(fn + '_A.csv')
            for idz, row in df.iterrows():
                #compute start and end frame position (quantizing)
                start = quantize(row['Start'])
                end = quantize(row['End'])
                start_frame = int(start)
                end_frame = int(end)
                class_id = self.dataset.label_dic[row['Class']]  #int ID of sound class name
                sound_frames = np.arange(start_frame, end_frame)
                x, y, z = row['X'], row['Y'], row['Z']
                azimuth = int(np.arctan2(y, x) * 180 / np.pi)
                elevation = int(np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi)
                for f in sound_frames:
                    local_frame_label = [class_id, idz, azimuth, elevation]
                    frame_label[f].append(local_frame_label)
            f = open(meta_path, 'w')
            for frame in range(num_frames):
                if frame_label[frame]:
                    for event in frame_label[frame]:
                        f.write('{},{},{},{},{}\n'.format(frame, event[0], event[1], event[2], event[3]))
            f.close()
            tqdm.write('{}, {}'.format(idy, meta_path))