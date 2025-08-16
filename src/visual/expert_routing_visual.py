# src/visual/timefreq_visual.py

from typing import List, Dict, Optional
import hydra
import lightning as L
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from tqdm import tqdm
import pickle
import sys
from omegaconf import DictConfig
import librosa
import warnings

warnings.filterwarnings('ignore')

# 修复路径问题 - 添加src目录到Python路径
current_dir = Path(__file__).parent
src_dir = current_dir.parent
project_root = src_dir.parent

# 添加src目录到路径，这样可以直接导入src下的模块
sys.path.insert(0, str(src_dir))

from utils.config import get_dataset
from utils.utilities import extras, get_pylogger

# 使用英文字体，避免中文字体兼容性问题
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

log = get_pylogger(__name__)

class TimeFreqExpertRoutingVisualizer:
    """时频域专家路由可视化工具类"""
    
    def __init__(self, cfg: DictConfig, model, datamodule):
        """
        初始化可视化工具
        
        Args:
            cfg: 配置对象
            model: 已加载的模型
            datamodule: 数据模块
        """
        self.cfg = cfg
        self.model = model
        self.datamodule = datamodule
        self.device = next(model.parameters()).device
        
        # 获取时频分析配置
        self.timefreq_cfg = cfg.visual.timefreq_analysis
        
        # 获取专家名称和router信息
        self.expert_names = self._get_expert_names_from_config()
        self.router_info = self._get_router_info()
        
        # 设置matplotlib和seaborn样式
        plt.style.use('default')
        sns.set_palette("husl")
        
    def _get_expert_names_from_config(self):
        """从配置中获取专家名称"""
        try:
            experts_config = self.cfg.adapt.adapt_kwargs.get('experts_config', [])
            names = []
            for expert in experts_config:
                names.append(expert.get('name', f'expert_{len(names)}'))
            return names
        except:
            # 使用配置文件中的备用专家名称
            try:
                return self.cfg.visual.timefreq_analysis.expert_detection.fallback_expert_names
            except:
                # 最后的默认专家名称
                return ['DCTAdapter', 'SEAdapter', 'LinearAdapter', 'ConvAdapter']
    
    def _get_router_info(self):
        """获取router模块信息"""
        router_info = {}
        
        def find_routers_recursive(module, path=""):
            routers = {}
            
            for name, child in module.named_children():
                current_path = f"{path}.{name}" if path else name
                
                # 查找MLP层中的router
                if 'mlp' in name.lower() and hasattr(child, 'adapter_instance'):
                    adapter = child.adapter_instance
                    if hasattr(adapter, 'router'):
                        routers[current_path] = {
                            'module': adapter.router,
                            'path': f"{current_path}.adapter_instance.router",
                            'depth': current_path.count('.')
                        }
                        log.info(f"找到Router模块: {current_path}.adapter_instance.router")
                
                # 递归查找子模块
                child_routers = find_routers_recursive(child, current_path)
                if child_routers:
                    routers.update(child_routers)
            
            return routers
        
        all_routers = find_routers_recursive(self.model.net)
        
        return all_routers
    
    def _get_dataloader(self):
        """获取数据加载器"""
        try:
            if hasattr(self.datamodule, 'val_set'):
                from torch.utils.data import DataLoader
                return DataLoader(
                    dataset=self.datamodule.val_set,
                    batch_size=1,  # 单样本处理便于可视化
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False
                )
            elif hasattr(self.datamodule, 'test_set'):
                from torch.utils.data import DataLoader
                return DataLoader(
                    dataset=self.datamodule.test_set,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False
                )
            else:
                raise AttributeError("数据模块中未找到可用的数据集")
        except Exception as e:
            log.error(f"创建数据加载器失败: {e}")
            raise

    def extract_expert_routing_data(self, max_samples=5):
        """
        提取专家路由数据和Log-Mel谱图
        
        Args:
            max_samples: 最多处理的样本数
            
        Returns:
            results: 包含原始音频、Log-Mel谱图和专家路由数据的结果列表
        """
        log.info("开始提取专家路由数据和时频特征...")
        
        results = []
        dataloader = self._get_dataloader()
        
        # 设置模型为评估模式
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, batch_sample in enumerate(tqdm(dataloader, desc="提取专家路由数据")):
                if batch_idx >= max_samples:
                    break
                
                batch_data = batch_sample['data'].to(self.device)
                batch_target = {key: value for key, value in batch_sample.items() if 'data' not in key}
                
                # 提取router输出（专家分配索引）
                routing_data = self._extract_routing_data(batch_data, batch_target)
                
                # 生成Log-Mel频谱图
                raw_audio = batch_data.detach().cpu().numpy()[0]  # [channels, time]
                log_mel_spec = self.generate_log_mel_spectrogram(raw_audio)
                
                # 保存当前样本的分析结果
                sample_result = {
                    'sample_idx': batch_idx,
                    'filename': batch_sample.get('filename', [f'sample_{batch_idx}'])[0],
                    'raw_audio': raw_audio,
                    'log_mel_spectrogram': log_mel_spec,
                    'routing_data': routing_data
                }
                
                results.append(sample_result)
                
                # 调试信息
                if routing_data:
                    for router_name, data in routing_data.items():
                        expert_indices = data['expert_indices']
                        log.info(f"样本 {batch_idx}, Router {router_name}: 专家分配索引形状 {expert_indices.shape}")
        
        log.info(f"成功提取了 {len(results)} 个样本的专家路由数据")
        return results
    
    def _extract_routing_data(self, batch_data, batch_target):
        """提取router的专家分配数据"""
        routing_data = {}
        hooks = []
        
        def create_routing_hook(router_name):
            def hook_fn(module, input, output):
                # router通常输出logits或概率分布
                if isinstance(output, torch.Tensor):
                    router_output = output.detach().cpu()
                elif isinstance(output, (list, tuple)) and len(output) > 0:
                    router_output = output[0].detach().cpu()
                else:
                    return output
                
                # 获取专家分配索引（argmax）
                expert_indices = torch.argmax(router_output, dim=-1)  # [B, H, W] or [B, T]
                
                # 获取路由概率（softmax）
                routing_probs = F.softmax(router_output, dim=-1)
                
                routing_data[router_name] = {
                    'expert_indices': expert_indices.numpy(),
                    'routing_probs': routing_probs.numpy(),
                    'raw_logits': router_output.numpy()
                }
                return output
            return hook_fn
        
        # 为router注册钩子
        for router_name, info in self.router_info.items():
            if info['module'] is not None:
                hook = info['module'].register_forward_hook(create_routing_hook(router_name))
                hooks.append(hook)
                log.info(f"为Router {router_name} 注册钩子")
        
        try:
            # 前向传播
            try:
                output = self.model.common_step(batch_data, batch_target)
            except Exception as e:
                log.warning(f"common_step失败，尝试直接调用网络: {e}")
                # 标准化输入
                if hasattr(self.model, 'standardize'):
                    batch_data_norm = self.model.standardize(batch_data)
                else:
                    batch_data_norm = batch_data
                output = self.model.net(batch_data_norm)
        
        finally:
            # 清理钩子
            for hook in hooks:
                hook.remove()
        # 只保留最后一层的router数据
        if routing_data:
            # 获取所有router名称
            router_names = list(routing_data.keys())
            if router_names:
                # 只保留最后一个router的数据
                last_router = router_names[-1] # 0 1 2 3 ..12. -1
                routing_data = {last_router: routing_data[last_router]}
                log.info(f"只保留最后一层Router数据: {last_router}")

        return routing_data
    
    def generate_log_mel_spectrogram(self, audio_data):
        """
        生成Log-Mel频谱图（使用配置参数）
        
        Args:
            audio_data: 音频数据 [channels, time]
            
        Returns:
            log_mel_spec: Log-Mel频谱图 [n_mels, time_frames]
        """
        # 获取配置参数
        mel_params = self.timefreq_cfg.mel_params
        sr = mel_params.sr
        n_mels = mel_params.n_mels
        n_fft = mel_params.n_fft
        hop_length = mel_params.hop_length
        fmax = mel_params.fmax
        
        # 如果是多通道，取平均
        if audio_data.ndim > 1:
            audio = np.mean(audio_data, axis=0)
        else:
            audio = audio_data
        
        # 计算Mel频谱图
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_mels=n_mels, 
            n_fft=n_fft, 
            hop_length=hop_length,
            fmax=fmax
        )
        
        # 转换为对数刻度
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel_spec

    def save_analysis_results(self, results, save_path):
        """保存分析结果"""
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        log.info(f"分析结果已保存到: {save_path}")
    
    def _overlay_expert_indices_uniform(self, ax, expert_indices, routing_params, log_mel_shape):
        """均匀分布专家索引在log-mel频谱图上"""
        
        # 如果expert_indices是多维的，先展平
        if expert_indices.ndim > 1:
            expert_indices_flat = expert_indices.flatten()
        else:
            expert_indices_flat = expert_indices
        
        total_indices = len(expert_indices_flat)
        H, W = log_mel_shape  # log-mel频谱图的形状
        
        log.info(f"专家索引总数: {total_indices}")
        log.info(f"Log-mel频谱图形状: {H} x {W}")
        
        # 获取配置参数
        expert_colors = routing_params.expert_colors
        font_size = routing_params.font_size
        alpha = routing_params.text_alpha
        
        # 计算均匀分布的网格
        # 根据专家索引数量确定网格大小
        if total_indices == 64:
            # 64个索引，可以用8x8网格
            grid_h, grid_w = 8, 8
        elif total_indices <= 16:
            # 16个或更少，用4x4网格
            grid_h, grid_w = 4, 4
        elif total_indices <= 36:
            # 36个或更少，用6x6网格
            grid_h, grid_w = 6, 6
        else:
            # 其他情况，计算最接近的正方形网格
            import math
            grid_size = int(math.ceil(math.sqrt(total_indices)))
            grid_h, grid_w = grid_size, grid_size
        
        log.info(f"使用网格大小: {grid_h} x {grid_w}")
        
        # 计算在频谱图上的位置
        step_h = H / grid_h  # 频率维度的步长
        step_w = W / grid_w  # 时间维度的步长
        
        # 均匀分布专家索引
        idx = 0
        for i in range(grid_h):
            for j in range(grid_w):
                if idx >= total_indices:
                    break
                
                # 计算在频谱图上的坐标
                y_pos = int(i * step_h + step_h / 2)  # 网格中心的y坐标
                x_pos = int(j * step_w + step_w / 2)  # 网格中心的x坐标
                
                # 确保坐标在有效范围内
                y_pos = min(y_pos, H - 1)
                x_pos = min(x_pos, W - 1)
                
                expert_idx = int(expert_indices_flat[idx])
                color = expert_colors[expert_idx % len(expert_colors)]
                
                # 绘制专家编号
                # ax.text(x_pos, y_pos, str(expert_idx), 
                #        color=color, fontsize=font_size, 
                #        ha='center', va='center', 
                #        alpha=alpha, fontweight='bold',
                #        bbox=dict(boxstyle="round,pad=0.3", 
                #                facecolor='white', alpha=0.8, 
                #                edgecolor=color, linewidth=2))
                
                # ax.text(
                #         x_pos, y_pos, str(expert_idx), 
                #         color=color, fontsize=8,  # 更小字体
                #         ha='center', va='center',
                #         alpha=0.7                 # 半透明
                #     )
                
                ax.text(
                        x_pos, y_pos, str(expert_idx), 
                        color=color, fontsize=font_size, 
                        ha='center', va='center', 
                        alpha=alpha, fontweight='bold'
                    )


                
                # 可选：添加网格线来显示分布区域
                # if routing_params.get('show_grid', False):
                #     # 绘制网格边界
                #     rect = plt.Rectangle((j * step_w, i * step_h), step_w, step_h,
                #                        fill=False, edgecolor=color, alpha=0.3, linewidth=1)
                #     ax.add_patch(rect)
                
                idx += 1
            
            if idx >= total_indices:
                break

    def create_uniform_expert_overlay_visualization(self, results, save_dir):
        """
        创建均匀分布的专家索引叠加可视化
        
        Args:
            results: 分析结果列表
            save_dir: 保存目录
        """
        log.info("生成均匀分布的专家索引叠加可视化...")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if not results:
            log.error("没有可用的分析结果")
            return
        
        # 获取可视化配置参数
        viz_params = self.timefreq_cfg.visualization_params
        routing_params = self.timefreq_cfg.routing_visualization
        
        for sample_idx, sample in enumerate(results):
            log_mel_spec = sample['log_mel_spectrogram']
            routing_data = sample['routing_data']
            
            if not routing_data:
                log.warning(f"样本 {sample_idx + 1} 没有路由数据")
                continue
            
            # 获取第一个（也是唯一的）router数据
            router_name, data = list(routing_data.items())[0]
            expert_indices = data['expert_indices']
            
            # 移除batch维度（如果存在）
            if expert_indices.ndim > 2:
                expert_indices = expert_indices[0]  # 取第一个batch
            elif expert_indices.ndim > 1 and expert_indices.shape[0] == 1:
                expert_indices = expert_indices[0]  # 移除batch维度
            
            log.info(f"处理后的专家索引形状: {expert_indices.shape}")
            
            # 创建可视化图
            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            
            # 显示Log-Mel作为背景 cmap='viridis' 'hot' 'jet' 'plasma'
            im = ax.imshow(log_mel_spec, aspect='auto', origin='lower', 
                          cmap='viridis', alpha=routing_params.background_alpha)
            
            # 均匀分布叠加专家索引
            self._overlay_expert_indices_uniform(ax, expert_indices, routing_params, log_mel_spec.shape)
            
            # 设置标题和标签
            ax.set_title(f'Sample {sample_idx + 1}\n'
                        f'File: {sample["filename"]}\n'
                        f'Router: {router_name} | Total Experts: 4', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Time Frames', fontsize=12)
            ax.set_ylabel('Mel Frequency Bins', fontsize=12)
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, shrink=0.8, label='Log-Mel Magnitude (dB)')
            
            # 添加专家统计信息（确保包含所有4个专家）
            all_experts = [0, 1, 2, 3]  # 确保包含所有专家
            unique_experts, counts = np.unique(expert_indices, return_counts=True)
            
            # 创建完整的专家统计字典
            expert_stats = {expert: 0 for expert in all_experts}
            for exp_idx, count in zip(unique_experts, counts):
                expert_stats[exp_idx] = count
            
            stats_text = f"Expert Usage:\n"
            total_patches = len(expert_indices.flatten())
            for expert_idx in all_experts:
                count = expert_stats[expert_idx]
                percentage = count / total_patches * 100
                stats_text += f"Expert {expert_idx}: {count} ({percentage:.1f}%)\n"
            
            # 在图上添加统计信息
            # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            #        fontsize=10, verticalalignment='top',
            #        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # 保存图片
            save_path = save_dir / f'sample_{sample_idx + 1}_uniform_expert_overlay.png'
            plt.savefig(save_path, dpi=viz_params.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
            
            log.info(f"样本 {sample_idx + 1} 的均匀专家分布图已保存到: {save_path}")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="expert_routing_visual.yaml")
def main(cfg: DictConfig):
    """专家路由可视化主函数"""
    
    # 应用额外的工具函数
    extras(cfg)

    # 设置随机种子
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
   
    # 确保使用验证模式
    if not hasattr(cfg, 'mode'):
        cfg.mode = 'valid'
    
    if cfg.mode == 'valid':
        # 验证模式
        default_dataset = list(cfg.data.valid_dataset.keys())[0]
        dataset = get_dataset(dataset_name=default_dataset, cfg=cfg)

        log.info(f"正在实例化数据模块 <{cfg.datamodule._target_}> ...")
        datamodule = hydra.utils.instantiate(cfg.datamodule, cfg, dataset, 'fit')
        valid_meta = datamodule.paths_dict, datamodule.valid_gt_dcaseformat

        log.info(f"正在实例化模型 <{cfg.modelmodule._target_}> ...")
        model = hydra.utils.instantiate(cfg.modelmodule, cfg, dataset, valid_meta)
        
        log.info("正在设置模型...")
        model.setup('valid')

    elif cfg.mode == 'test':
        # 测试模式
        default_dataset = list(cfg.data.test_dataset.keys())[0]
        dataset = get_dataset(dataset_name=default_dataset, cfg=cfg)

        log.info(f"正在实例化数据模块 <{cfg.datamodule._target_}> ...")
        datamodule = hydra.utils.instantiate(cfg.datamodule, cfg, dataset, 'test')
        test_meta = datamodule.paths_dict

        log.info(f"正在实例化模型 <{cfg.modelmodule._target_}> ...")
        model = hydra.utils.instantiate(cfg.modelmodule, cfg, dataset, test_meta=test_meta)
        
        log.info("正在设置模型...")
        model.setup('test')

    # 加载模型权重
    if cfg.get("ckpt_path"):
        log.info(f"正在加载模型权重: {cfg.ckpt_path}")
        checkpoint = torch.load(cfg.ckpt_path, map_location='cpu', weights_only=True)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=True)
    
    # 移动模型到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 保存结果
    save_dir = Path(cfg.paths.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 创建专家路由可视化器
    log.info("正在创建专家路由可视化器...")
    visualizer = TimeFreqExpertRoutingVisualizer(cfg, model, datamodule)
    
    # 提取专家路由数据
    log.info("开始提取专家路由数据...")
    results = visualizer.extract_expert_routing_data(max_samples=cfg.visual.num_samples)
    
    if not results:
        log.error("未提取到任何专家路由数据，退出可视化")
        return
    
    # 创建均匀分布的专家叠加可视化（主要功能）
    log.info("正在生成均匀分布的专家叠加可视化...")
    visualizer.create_uniform_expert_overlay_visualization(results, save_dir)
    
    log.info("专家路由可视化完成！")
    log.info(f"所有结果已保存到: {save_dir}")
  
if __name__ == "__main__":
    # 设置矩阵乘法精度
    torch.set_float32_matmul_precision('medium')
    
    # 执行主函数
    main() 





    