# 导入所需的库和模块
from pathlib import Path  # 用于路径处理
import logging  # 用于日志记录

import models  # 导入模型包
from models.components.model_module import BaseModelModule  # 导入基础模型模块
from utils.data_utilities import write_output_format_file  # 导入输出格式化工具

import torch  # PyTorch深度学习库
import numpy as np  # 科学计算库
from tqdm import tqdm  # 进度条显示
import torch.nn.functional as F  # PyTorch函数式API

# 模型模块字典，用于根据配置动态选择模型类型
ModelMoodule = {
    'accdoa': models.accdoa,  # 标准ACCDOA模型
    'einv2': models.einv2,    # EINV2模型
    'multi_accdoa': models.multi_accdoa,  # 支持多轨道的ACCDOA模型
}


class SELDModelModule(BaseModelModule):
    # 声音事件定位与检测(SELD)模型模块，继承自BaseModelModule
    
    def setup(self, stage):
        # 设置模型，根据当前阶段初始化网络

        audio_feature = self.cfg.data.audio_feature  # 获取音频特征类型
        kwargs = self.cfg.model.kwargs  # 获取模型参数
        # 根据不同的音频特征类型设置输入通道数
        if audio_feature in ['logmelIV', 'salsa', 'salsalite']:
            in_channels = 7  # 对于强度向量特征，使用7个通道
        elif audio_feature in ['logmelgcc']:
            in_channels = 10  # 对于GCC特征，使用10个通道
        elif audio_feature in ['logmel']:
            in_channels = 1  # 对于单纯的对数梅尔频谱，使用1个通道
        
        # 实例化网络模型
        self.net = vars(ModelMoodule[self.method])[self.cfg.model.backbone](
            self.cfg, self.num_classes, in_channels, **kwargs)
        if self.cfg.compile:
            self.logging.info('Compiling model')  # 记录编译模型信息
            self.net = torch.compile(self.net)  # 使用PyTorch 2.0编译功能加速模型

        # 记录模型参数数量
        self.logging.info("Number of parameters of net: " + 
                            f"{sum(p.numel() for p in self.net.parameters())}")
        if stage == 'test':
            # 测试阶段，创建提交结果的目录
            logger = logging.getLogger()
            log_filename = logger.handlers[1].baseFilename
            self.submissions_dir = Path(log_filename).parent / 'submissions'
            self.submissions_dir.mkdir(exist_ok=True)
    
    def common_step(self, batch_x, batch_y=None):
        # 模型前向传播的通用步骤，包含数据增强和标准化处理

        # 特征提取前的数据增强
        if self.training:
            if self.data_aug['AugMix']: 
                batch_x, batch_y = self.data_copy(batch_x, batch_y)  # 复制数据用于AugMix
            if 'rotate' in self.data_aug['type']:
                batch_x, batch_y = self.data_aug['rotate'](batch_x, batch_y)  # 应用旋转增强
            if 'wavmix' in self.data_aug['type']:
                batch_x, batch_y = self.data_aug['wavmix'](batch_x, batch_y)  # 应用波形混合增强
        
        batch_x = self.standardize(batch_x)  # 标准化输入特征 [B, C, T]  C 表示通道数, T 表示时间序列长度（与音频采样率有关）
        
        # 特征提取后的数据增强
        if self.training:
            if self.data_aug['AugMix']:
                batch_x, batch_y = self.augmix_data(batch_x, batch_y)  # 应用AugMix增强
            else:
                batch_x, batch_y = self.augment_data(batch_x, batch_y)  # 应用标准数据增强

        batch_x = self.forward(batch_x)  # 模型前向传播，HTSAT模型特征处理
        return batch_x, batch_y  # 返回处理后的数据和标签

    def training_step(self, batch_sample, batch_idx):
        # 训练步骤，处理一个批次的数据并计算损失
        
        # 统计不同重叠度的样本数量
        self.stat['ov1'] += np.sum(np.array(batch_sample['ov']) == '1')  # 重叠度1
        self.stat['ov2'] += np.sum(np.array(batch_sample['ov']) == '2')  # 重叠度2
        self.stat['ov3'] += np.sum(np.array(batch_sample['ov']) == '3')  # 重叠度3
        batch_data = batch_sample['data']  # 获取输入数据 [B, C, T] C 表示通道数, T 表示时间序列长度（与音频采样率有关）
        # 获取目标数据 [B, T, CSE, SI, C]  T表示时间帧数，CSE表示最大同时发生事件数，SI表示空间信息维度（方向角、仰角、距离、类索引），C表示事件类别数
        batch_target = {key: value for key, value in batch_sample.items() if 'data' not in key}  # 获取目标数据
        batch_pred, batch_target = self.common_step(batch_data, batch_target)  # 通用步骤处理
        
        # 计算主损失
        loss_dict = self.loss(batch_pred, batch_target)  # 计算损失
        loss_dict[self.loss.loss_type] = loss_dict[self.loss.loss_type]  # 确保主损失存在
        
        # 收集并添加辅助损失
        aux_loss = 0.0
        for module in self.net.modules():
            if hasattr(module, 'aux_loss'):
                aux_loss += module.aux_loss
                # print(module)
                # print(f"aux_loss: {aux_loss}")

        # 将辅助损失添加到主损失中
        total_loss = loss_dict[self.loss.loss_type] + aux_loss

        # 更新训练损失字典
        for key in loss_dict.keys():
            self.train_loss_dict[key].update(loss_dict[key])
        
        # return loss_dict[self.loss.loss_type]  # 返回主损失用于反向传播
        return total_loss  # 返回总损失用于反向传播

    def validation_step(self, batch_sample, batch_idx):
        # 验证步骤，处理一个批次的验证数据
        batch_data = batch_sample['data']  # 获取输入数据
        batch_target = {key: value for key, value in batch_sample.items()
                        if 'label' in key}  # 获取标签数据
        # 根据配置选择是否应用后处理
        if self.cfg.get('post_processing') == 'ACS':
            # 使用坐标变换和符号变换来增强预测结果
            batch_pred = self.post_processing(batch_data, method='ACS', output_format=self.method)
        else:
            batch_pred = self.common_step(batch_data)[0]  # 通过通用步骤获取预测结果

        self.step_system_outputs.append(batch_pred)  # 保存预测结果
        loss_dict = self.loss(batch_pred, batch_target)  # 计算验证损失
        for key in loss_dict.keys():
            self.val_loss_dict[key].update(loss_dict[key])  # 更新验证损失字典

    def on_validation_epoch_start(self):
        # 验证轮开始时的回调函数
        
        print(self.stat)  # 打印当前统计信息
        self.stat = {'ov1': 0, 'ov2': 0, 'ov3': 0}  # 重置统计计数器
        self.metrics.reset()  # 重置度量指标
    
    def on_load_checkpoint(self, checkpoint):
        # 加载检查点时的回调函数
        
        if self.cfg.compile:
            return  # 如果启用编译模式，跳过处理
        # 处理由torch.compile生成的特殊前缀
        keys_list = list(checkpoint['state_dict'].keys())
        for key in keys_list:
            if 'orig_mod.' in key:
                deal_key = key.replace('_orig_mod.', '')
                checkpoint['state_dict'][deal_key] = checkpoint['state_dict'][key]
                del checkpoint['state_dict'][key]

    def on_validation_epoch_end(self):
        # 验证轮结束时的回调函数
        
        pred_sed, pred_doa = self.pred_aggregation()  # 聚合预测结果
        
        ######## 计算度量指标 ########
        frame_ind = 0
        paths = tqdm(self.valid_paths_dict.keys(), 
                     desc='Computing metrics for validation set')  # 使用tqdm显示进度
        f = open('metrics.csv', 'w')  # 打开文件用于保存度量结果
        for path in paths:
            loc_frames = self.valid_paths_dict[path]  # 获取当前路径的帧数
            num_frames = self.get_num_frames(loc_frames)  # 获取帧数
            # 将预测结果转换为DCASE评估格式（极坐标）
            pred_dcase_format = self.convert_to_dcase_format_polar(
                pred_sed=pred_sed[frame_ind:frame_ind+loc_frames],
                pred_doa=pred_doa[frame_ind:frame_ind+loc_frames])
            gt_dcase_format = self.valid_gt_dcase_format[path]  # 获取真实标签
            # 更新度量指标
            self.update_metrics(
                pred_dcase_format=pred_dcase_format, 
                gt_dcase_format=gt_dcase_format, 
                num_frames=loc_frames)
            frame_ind += num_frames
        f.close()  # 关闭文件

        ######## 记录日志 ########
        self.logging.info("-------------------------------------------"
                 + "---------------------------------------")
        # 计算并记录宏平均度量指标
        metric_dict, _ = self.metrics.compute_seld_scores(average='macro')
        self.log_metrics(metric_dict, set_type='val/macro')
        # 计算并记录微平均度量指标
        metric_dict, _ = self.metrics.compute_seld_scores(average='micro')
        self.log_metrics(metric_dict, set_type='val/micro')
        # 记录验证损失
        self.log_losses(self.val_loss_dict, set_type='val')

    
    def on_train_epoch_end(self):
        # 训练轮结束时的回调函数
        
        # 获取当前学习率和最大轮数
        lr = self.optimizers().param_groups[0]['lr']
        max_epochs = self.cfg.trainer.max_epochs
        # 记录训练损失
        self.log_losses(self.train_loss_dict, set_type='train')
        self.log('lr', lr)  # 记录学习率
        # 输出当前训练进度和学习率
        self.logging.info(f"Epoch/Total Epoch: {self.current_epoch+1}/{max_epochs}, LR: {lr}")
        self.logging.info("-------------------------------------------"
                 + "---------------------------------------")

    def on_test_epoch_start(self):
        # 测试轮开始时的回调函数
        
        self.step_system_outputs = []  # 初始化系统输出列表
    
    def test_step(self, batch_sample, batch_idx):
        # 测试步骤，处理一个批次的测试数据
        
        batch_data = batch_sample['data']  # 获取输入数据
        batch_pred = self.common_step(batch_data)[0]  # 通过通用步骤获取预测结果
        self.step_system_outputs.append(batch_pred)  # 保存预测结果
    
    def on_test_epoch_end(self):
        # 测试轮结束时的回调函数
        
        pred_sed, pred_doa = self.pred_aggregation()  # 聚合预测结果
        
        # 处理每个测试样本
        frame_ind = 0
        for path in self.test_paths_dict.keys():
            loc_frames = self.test_paths_dict[path]  # 获取当前路径的帧数
            fn = Path(path).stem + '.csv'  # 生成输出文件名
            num_frames = self.get_num_frames(loc_frames)  # 获取帧数
            # 将预测结果转换为DCASE评估格式（极坐标）
            pred_dcase_format = self.convert_to_dcase_format_polar(
                pred_sed=pred_sed[frame_ind:frame_ind+loc_frames],
                pred_doa=pred_doa[frame_ind:frame_ind+loc_frames])
            csv_path = self.submissions_dir.joinpath(fn)  # 构建输出文件路径
            # 将结果写入CSV文件
            write_output_format_file(csv_path, pred_dcase_format)
            frame_ind += num_frames
        # 输出结果保存位置
        self.logging.info('Rsults are saved to {}\n'.format(str(self.submissions_dir)))