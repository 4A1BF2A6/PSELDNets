from typing import List, Dict, Optional
import hydra
import lightning as L
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
import cv2
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

# 设置英文字体，避免中文字体问题
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

log = get_pylogger(__name__)

class MOEExpertGradCAMVisualizer:
    """MOE专家Grad-CAM可视化工具类"""

    def __init__(self, cfg: DictConfig, model, datamodule):
        """
        初始化可视化器，保存配置、模型、数据模块，并自动注册专家模块。

        参数:
            cfg: 配置对象
            model: 已加载的模型
            datamodule: 数据模块
        """
        self.cfg = cfg
        self.model = model
        self.datamodule = datamodule
        self.device = next(model.parameters()).device

        # 获取专家名称列表
        self.expert_names = self._get_expert_names()

        # 用于存储每个专家的特征和梯度
        self.expert_features = {}
        self.expert_gradients = {}

        # 专家模块映射字典
        self.expert_modules = {}
        self._register_expert_modules()

        log.info(f"初始化MOE Grad-CAM可视化工具，发现 {len(self.expert_names)} 个专家")

    def _get_expert_names(self):
        """
        从配置文件中获取专家名称列表。

        返回:
            names: 专家名称列表
        """
        try:
            experts_config = self.cfg.adapt.adapt_kwargs.get('experts_config', [])
            names = []
            for expert in experts_config:
                names.append(expert.get('name', f'expert_{len(names)}'))
            return names
        except:
            # 如果配置中没有，使用备用专家名称
            try:
                return self.cfg.visual.gradcam_analysis.expert_detection.fallback_expert_names
            except:
                return ['DCTAdapter', 'SEAdapter', 'LinearAdapter', 'ConvAdapter']

    def _register_expert_modules(self):
        """
        只注册最后一层MLP后的MOE专家。
        递归遍历模型，找到所有路径包含'mlp'的MOE模块，选出最深（最后一层）的那一个，只注册这组专家。
        """
        mlp_expert_modules = []

        def find_expert_modules(module, path=""):
            for name, child in module.named_children():
                current_path = f"{path}.{name}" if path else name
                # 只关注路径中包含'mlp'的模块
                if 'mlp' in current_path and hasattr(child, 'experts') and hasattr(child, 'expert_names'):
                    mlp_expert_modules.append((current_path, child))
                # 递归查找
                find_expert_modules(child, current_path)

        # 开始查找
        find_expert_modules(self.model.net)

        # 找到最深的MLP模块
        if mlp_expert_modules:
            # 按照路径中的层数进行排序，选择最后一层
            # deepest_mlp_path, deepest_mlp_module = max(mlp_expert_modules, key=lambda x: int(x[0].split('.')[2]))
            deepest_mlp_path, deepest_mlp_module = mlp_expert_modules[-1]
            for i, expert in enumerate(deepest_mlp_module.experts):
                expert_name = deepest_mlp_module.expert_names[i] if i < len(deepest_mlp_module.expert_names) else f"expert_{i}"
                if expert_name in self.expert_names:
                    self.expert_modules[expert_name] = expert
                    log.info(f"注册最后一层MLP专家模块: {expert_name}")

    def _get_dataloader(self):
        """
        获取数据加载器（DataLoader），用于遍历验证集或测试集。

        返回:
            DataLoader对象
        """
        try:
            if hasattr(self.datamodule, 'val_set'):
                from torch.utils.data import DataLoader
                return DataLoader(
                    dataset=self.datamodule.val_set,
                    batch_size=1,  # 单样本处理，便于可视化
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

    def generate_expert_gradcam(self, max_samples=5, save_dir='./gradcam_results'):
        """
        批量生成专家Grad-CAM可视化图片。

        参数:
            max_samples: 最多处理多少个样本
            save_dir: 图片保存目录
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        dataloader = self._get_dataloader()
        log.info("开始生成MOE专家Grad-CAM可视化...")

        # 遍历每个样本
        for batch_idx, batch_sample in enumerate(tqdm(dataloader, desc="生成Grad-CAM")):
            if batch_idx >= max_samples:
                break
            # 生成当前样本的专家Grad-CAM
            gradcam_results = self._generate_sample_gradcam(batch_sample, batch_idx)
            # 可视化并保存结果
            self._visualize_and_save_gradcam(gradcam_results, batch_idx, save_dir, batch_sample)
        log.info(f"Grad-CAM可视化完成，结果保存在: {save_dir}")

    def _generate_sample_gradcam(self, batch_sample, sample_idx):
        """
        对单个样本生成所有专家的Grad-CAM热力图。

        参数:
            batch_sample: 当前批次的数据（字典）
            sample_idx: 当前样本编号

        返回:
            gradcam_results: 包含原始频谱图、每个专家热力图等信息的字典
        """
        batch_data = batch_sample['data'].to(self.device)
        batch_target = {key: value for key, value in batch_sample.items() if 'data' not in key}

        # 清空之前的特征和梯度
        self.expert_features.clear()
        self.expert_gradients.clear()

        # 注册hook，提取特征和梯度
        hooks = self._register_hooks()

        try:
            # 设置为训练模式，便于反向传播
            self.model.train()
            batch_data.requires_grad_(True)
            # 标准化输入（如果有）
            if hasattr(self.model, 'standardize'):
                batch_data_norm = self.model.standardize(batch_data)
            else:
                batch_data_norm = batch_data
            # 前向传播
            output = self.model.net(batch_data_norm)
            # 计算每个专家的Grad-CAM
            expert_gradcams = {}
            for expert_name in self.expert_names:
                if expert_name in self.expert_features:
                    # 取输出最大值作为目标
                    if isinstance(output, dict):
                        target_output = output.get('accdoa', output.get('sed', list(output.values())[0]))
                    else:
                        target_output = output
                    target_score = target_output.max()
                    # 反向传播，计算梯度
                    self.model.zero_grad()
                    if target_score.requires_grad:
                        target_score.backward(retain_graph=True)
                    # 计算Grad-CAM热力图
                    gradcam_map = self._compute_gradcam(expert_name)
                    if gradcam_map is not None:
                        expert_gradcams[expert_name] = gradcam_map
            # 生成原始频谱图
            raw_audio = batch_data.detach().cpu().numpy()[0]
            original_spectrogram = self._generate_mel_spectrogram(raw_audio)
            return {
                'sample_idx': sample_idx,
                'filename': batch_sample.get('filename', [f'sample_{sample_idx}'])[0],
                'original_spectrogram': original_spectrogram,
                'expert_gradcams': expert_gradcams,
                'raw_audio': raw_audio
            }
        finally:
            # 清理hook，防止内存泄漏
            for hook in hooks:
                hook.remove()
            self.model.eval()

    def _register_hooks(self):
        """
        为每个专家注册前向和反向hook，用于提取特征和梯度。

        返回:
            hooks: 所有注册的hook对象列表
        """
        hooks = []
        def forward_hook_fn(expert_name):
            def hook(module, input, output):
                # 存储前向输出特征
                if isinstance(output, torch.Tensor):
                    self.expert_features[expert_name] = output.clone()
            return hook
        def backward_hook_fn(expert_name):
            def hook(module, grad_input, grad_output):
                # 存储反向传播的梯度
                if grad_output[0] is not None:
                    self.expert_gradients[expert_name] = grad_output[0].clone()
            return hook
        # 为每个专家注册hook
        for expert_name, expert_module in self.expert_modules.items():
            forward_hook = expert_module.register_forward_hook(forward_hook_fn(expert_name))
            hooks.append(forward_hook)
            backward_hook = expert_module.register_backward_hook(backward_hook_fn(expert_name))
            hooks.append(backward_hook)
        return hooks

    def _compute_gradcam(self, expert_name):
        """
        计算单个专家的Grad-CAM热力图。

        参数:
            expert_name: 专家名称

        返回:
            gradcam: 归一化后的热力图（二维numpy数组）
        """
        if expert_name not in self.expert_features or expert_name not in self.expert_gradients:
            log.warning(f"专家 {expert_name} 缺少特征或梯度信息")
            return None
        features = self.expert_features[expert_name]  # [B, N, C] 或 [B, C, H, W]
        gradients = self.expert_gradients[expert_name]  # 同上
        # 计算通道权重（全局平均池化）
        if gradients.dim() == 3:  # [B, N, C]，Transformer特征
            weights = torch.mean(gradients, dim=(0, 1))  # [C]
            weighted_features = features * weights.unsqueeze(0).unsqueeze(0)  # [B, N, C]
            gradcam = torch.sum(weighted_features, dim=2)  # [B, N]
            gradcam = gradcam[0]  # [N]
            gradcam = self._reshape_sequence_to_2d(gradcam)
        elif gradients.dim() == 4:  # [B, C, H, W]，CNN特征
            weights = torch.mean(gradients, dim=(0, 2, 3))  # [C]
            weighted_features = features * weights.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            gradcam = torch.sum(weighted_features, dim=1)  # [B, H, W]
            gradcam = gradcam[0]  # [H, W]
        else:
            log.warning(f"不支持的特征维度: {gradients.dim()}")
            return None
        # 只保留正值
        gradcam = F.relu(gradcam)
        # 归一化到[0, 1]
        if gradcam.max() > gradcam.min():
            gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())
        return gradcam.detach().cpu().numpy()

    def _reshape_sequence_to_2d(self, sequence_tensor):
        """
        将一维序列形式的tensor重塑为二维（如8x8、16x16等）。

        参数:
            sequence_tensor: 一维tensor

        返回:
            二维tensor
        """
        N = sequence_tensor.shape[0]
        H = W = int(np.sqrt(N))
        if H * W != N:
            # 不是完全平方数，补齐
            H = int(np.sqrt(N))
            W = N // H
            if H * W < N:
                H += 1
            if H * W > N:
                padding = H * W - N
                sequence_tensor = F.pad(sequence_tensor, (0, padding))
        return sequence_tensor.view(H, W)

    def _generate_mel_spectrogram(self, audio_data):
        """
        生成Log-Mel频谱图。

        参数:
            audio_data: 原始音频数据（numpy数组）

        返回:
            log_mel_spec: Log-Mel频谱图（二维数组）
        """
        mel_params = self.cfg.visual.gradcam_analysis.mel_params
        sr = mel_params.sr
        n_mels = mel_params.n_mels
        n_fft = mel_params.n_fft
        hop_length = mel_params.hop_length
        fmax = mel_params.fmax
        # 多通道音频取平均
        if audio_data.ndim > 1:
            audio = np.mean(audio_data, axis=0)
        else:
            audio = audio_data
        # 计算Mel谱
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmax=fmax
        )
        # 转为对数刻度
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec

    def _visualize_and_save_gradcam(self, gradcam_results, sample_idx, save_dir, batch_sample):
        """
        可视化并保存Grad-CAM结果（详细中文注释版）

        参数说明：
        gradcam_results: 当前样本的Grad-CAM结果字典，包含原始频谱图和每个专家的热力图
        sample_idx: 当前样本编号
        save_dir: 图片保存目录
        batch_sample: 当前批次的原始数据（可用于获取文件名等信息）
        """
        original_spec = gradcam_results['original_spectrogram']  # 原始Log-Mel频谱图
        expert_gradcams = gradcam_results['expert_gradcams']     # 每个专家的Grad-CAM热力图
        filename = gradcam_results['filename']                   # 文件名

        # 获取可视化参数（如透明度、分辨率等）
        viz_params = self.cfg.visual.gradcam_analysis.visualization_params

        n_experts = len(expert_gradcams)  # 专家数量
        if n_experts == 0:
            log.warning(f"Sample {sample_idx} has no valid Grad-CAM results")
            return

        # 创建2行(n_experts+1)列的子图
        # fig, axes = plt.subplots(2, n_experts + 1, figsize=(4 * (n_experts + 1), 8))

        # 第一行：原始频谱图 + 每个专家的Grad-CAM热力图
        # im0 = axes[0, 0].imshow(original_spec, aspect='auto', origin='lower', cmap='viridis')
        # axes[0, 0].set_title('Original Log-Mel Spectrogram', fontsize=12, fontweight='bold')
        # axes[0, 0].set_xlabel('Time Frames')
        # axes[0, 0].set_ylabel('Mel Frequency')
        # plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)

        # for i, (expert_name, gradcam) in enumerate(expert_gradcams.items()):
        #     ax = axes[0, i + 1]
        #     im = ax.imshow(gradcam, aspect='auto', origin='lower', cmap='jet', alpha=0.8)
        #     ax.set_title(f'{expert_name}\nGrad-CAM Heatmap', fontsize=10, fontweight='bold')
        #     ax.set_xlabel('Time Frames')
        #     ax.set_ylabel('Frequency')
        #     plt.colorbar(im, ax=ax, shrink=0.8)

        # 第二行：原始频谱图与Grad-CAM热力图叠加
        # axes[1, 0].imshow(original_spec, aspect='auto', origin='lower', cmap='viridis')
        # axes[1, 0].set_title('Reference: Original Spectrogram', fontsize=12, fontweight='bold')
        # axes[1, 0].set_xlabel('Time Frames')
        # axes[1, 0].set_ylabel('Mel Frequency')

        # 只保留第二行
        fig, axes = plt.subplots(1, n_experts + 1, figsize=(4 * (n_experts + 1), 4))
        im0 = axes[0].imshow(original_spec, aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title('(a) Log-Mel Spectrogram', fontsize=15, )
        axes[0].set_xlabel('Time Frames')
        axes[0].set_ylabel('Mel Frequency')
        plt.colorbar(im0, ax=axes[0], shrink=0.8)
        
        for i, (expert_name, gradcam) in enumerate(expert_gradcams.items()):
            # ax = axes[1, i + 1]
            ax = axes[i + 1]
            ax.imshow(original_spec, aspect='auto', origin='lower', cmap='viridis', alpha=viz_params.background_alpha)
            if gradcam.shape != original_spec.shape:
                gradcam_resized = cv2.resize(gradcam, (original_spec.shape[1], original_spec.shape[0]))
            else:
                gradcam_resized = gradcam
            im_overlay = ax.imshow(gradcam_resized, aspect='auto', origin='lower', 
                                   cmap=viz_params.colormap, alpha=viz_params.overlay_alpha, vmin=0, vmax=1)
            # ax.set_title(f'{expert_name}\nOverlay Visualization', fontsize=10, fontweight='bold')
            if expert_name == 'dct_expert':
                expert_name = '(b) DCTAdapter'
            elif expert_name == 'SE_expert':
                expert_name = '(c) SEAdapter'
            elif expert_name == 'base_expert_1':
                expert_name = '(d) LinearAdapter'
            elif expert_name == 'mona_expert':
                expert_name = '(e) ConvAdapter'

            ax.set_title(f'{expert_name}', fontsize=15, )
            ax.set_xlabel('Time Frames')
            ax.set_ylabel('Frequency')
            plt.colorbar(im_overlay, ax=ax, shrink=0.8, ) # label='Attention Intensity'

        # fig.suptitle(f'Sample {sample_idx}: {filename}\nMOE Expert Grad-CAM Visualization', 
        #              fontsize=14, fontweight='bold')
        fig.suptitle(f'Mel-spectrogram and corresponding feature maps from \naudio segment of the people talking.', fontsize=15, )  
        
        # 在图像下方中央添加一行文字说明
        # fig.text(0.5, -0.05, "Mel-spectrogram and corresponding Grad-CAM feature maps derived from a sample of the xxx class.", ha='center', va='center', fontsize=14)
        
        # 设置总标题
        fig.suptitle(f'Mel-spectrogram and corresponding feature maps from \naudio segment of the people talking.', fontsize=15)

        plt.tight_layout()
        save_path = save_dir / f'gradcam_sample_{sample_idx}.png'
        plt.savefig(save_path, dpi=viz_params.dpi, bbox_inches='tight')
        plt.close()
        log.info(f"Sample {sample_idx} Grad-CAM visualization saved: {save_path}")

        # 保存数值结果（可用于后续分析）
        # results_path = save_dir / f'gradcam_data_sample_{sample_idx}.pkl'
        # with open(results_path, 'wb') as f:
        #     pickle.dump(gradcam_results, f)

    def generate_expert_comparison_heatmap(self, max_samples=10, save_path='./expert_comparison_heatmap.png'):
        """
        生成专家对比热力图，展示不同专家在不同样本上的平均注意力强度。

        参数:
            max_samples: 处理的样本数
            save_path: 图片保存路径
        """
        log.info("生成专家对比热力图...")
        dataloader = self._get_dataloader()
        all_expert_attention = {name: [] for name in self.expert_names}
        for batch_idx, batch_sample in enumerate(tqdm(dataloader, desc="收集专家注意力数据")):
            if batch_idx >= max_samples:
                break
            gradcam_results = self._generate_sample_gradcam(batch_sample, batch_idx)
            for expert_name, gradcam in gradcam_results['expert_gradcams'].items():
                avg_attention = np.mean(gradcam)
                all_expert_attention[expert_name].append(avg_attention)
        expert_names = list(all_expert_attention.keys())
        attention_matrix = np.array([all_expert_attention[name] for name in expert_names])
        plt.figure(figsize=(12, 8))
        sns.heatmap(attention_matrix, 
                   xticklabels=[f'Sample{i+1}' for i in range(attention_matrix.shape[1])],
                   yticklabels=expert_names,
                   cmap='YlOrRd', 
                   annot=True, 
                   fmt='.3f',
                   cbar_kws={'label': 'Average Attention Intensity'})
        plt.title('MOE Expert Attention Intensity Comparison Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Sample', fontsize=12)
        plt.ylabel('Expert', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        log.info(f"专家对比热力图已保存: {save_path}")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="moe_gradcam_visual.yaml")
def main(cfg: DictConfig):
    """MOE Expert Grad-CAM Visualization Main Function"""
    
    # Apply additional utility functions
    extras(cfg)

    # Set random seed
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
   
    # Ensure using validation mode
    if not hasattr(cfg, 'mode'):
        cfg.mode = 'valid'
    
    if cfg.mode == 'valid':
        # Validation mode
        default_dataset = list(cfg.data.valid_dataset.keys())[0]
        dataset = get_dataset(dataset_name=default_dataset, cfg=cfg)

        log.info(f"Instantiating data module <{cfg.datamodule._target_}> ...")
        datamodule = hydra.utils.instantiate(cfg.datamodule, cfg, dataset, 'fit')
        valid_meta = datamodule.paths_dict, datamodule.valid_gt_dcaseformat

        log.info(f"Instantiating model <{cfg.modelmodule._target_}> ...")
        model = hydra.utils.instantiate(cfg.modelmodule, cfg, dataset, valid_meta)
        
        log.info("Setting up model...")
        model.setup('valid')

    elif cfg.mode == 'test':
        # Test mode
        default_dataset = list(cfg.data.test_dataset.keys())[0]
        dataset = get_dataset(dataset_name=default_dataset, cfg=cfg)

        log.info(f"Instantiating data module <{cfg.datamodule._target_}> ...")
        datamodule = hydra.utils.instantiate(cfg.datamodule, cfg, dataset, 'test')
        test_meta = datamodule.paths_dict

        log.info(f"Instantiating model <{cfg.modelmodule._target_}> ...")
        model = hydra.utils.instantiate(cfg.modelmodule, cfg, dataset, test_meta=test_meta)
        
        log.info("Setting up model...")
        model.setup('test')

    # Load model weights
    if cfg.get("ckpt_path"):
        log.info(f"Loading model weights: {cfg.ckpt_path}")
        checkpoint = torch.load(cfg.ckpt_path, map_location='cpu', weights_only=True)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=True)
    
    # Move model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Save results
    save_dir = Path(cfg.paths.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create MOE Expert Grad-CAM visualizer
    log.info("Creating MOE Expert Grad-CAM visualizer...")
    visualizer = MOEExpertGradCAMVisualizer(cfg, model, datamodule)
    
    # Generate expert Grad-CAM visualizations
    log.info("Starting expert Grad-CAM visualization generation...")
    visualizer.generate_expert_gradcam(
        max_samples=cfg.visual.num_samples,
        save_dir=save_dir
    )
    
    # Generate expert comparison heatmap
    log.info("Generating expert comparison heatmap...")
    comparison_save_path = save_dir / "expert_gradcam_comparison_heatmap.png"
    visualizer.generate_expert_comparison_heatmap(
        max_samples=cfg.visual.num_samples,
        save_path=str(comparison_save_path)
    )
    
    log.info("MOE Expert Grad-CAM visualization completed!")
    log.info(f"All results saved to: {save_dir}")
  
if __name__ == "__main__":
    # Set matrix multiplication precision
    torch.set_float32_matmul_precision('medium')
    
    # Execute main function
    main() 