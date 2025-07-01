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
    """MOE Expert Grad-CAM Visualizer - Gradient-based Class Activation Mapping"""
    
    def __init__(self, cfg: DictConfig, model, datamodule):
        """
        Initialize MOE Expert Grad-CAM Visualizer
        
        Args:
            cfg: Configuration object
            model: Loaded model
            datamodule: Data module
        """
        self.cfg = cfg
        self.model = model
        self.datamodule = datamodule
        self.device = next(model.parameters()).device
        
        # Get expert names
        self.expert_names = self._get_expert_names()
        
        # Storage containers for features and gradients
        self.expert_features = {}  # Store expert features
        self.expert_gradients = {}  # Store expert gradients
        
        # Expert module mapping
        self.expert_modules = {}
        self._register_expert_modules()
        
        log.info(f"Initialized MOE Grad-CAM visualizer, found {len(self.expert_names)} experts")
        
    def _get_expert_names(self):
        """Get expert names from configuration"""
        try:
            experts_config = self.cfg.adapt.adapt_kwargs.get('experts_config', [])
            names = []
            for expert in experts_config:
                names.append(expert.get('name', f'expert_{len(names)}'))
            return names
        except:
            # Use fallback expert names from config file
            try:
                return self.cfg.visual.gradcam_analysis.expert_detection.fallback_expert_names
            except:
                return ['dct_expert', 'SE_expert', 'base_expert', 'mona_expert']
    
    def _register_expert_modules(self):
        """Register expert modules for hooks"""
        def find_expert_modules(module, path=""):
            for name, child in module.named_children():
                current_path = f"{path}.{name}" if path else name
                
                # Find mixture adapter modules
                if hasattr(child, 'experts') and hasattr(child, 'expert_names'):
                    for i, expert in enumerate(child.experts):
                        expert_name = child.expert_names[i] if i < len(child.expert_names) else f"expert_{i}"
                        if expert_name in self.expert_names:
                            self.expert_modules[expert_name] = expert
                            log.info(f"Registered expert module: {expert_name}")
                
                find_expert_modules(child, current_path)
        
        find_expert_modules(self.model.net)
    
    def _get_dataloader(self):
        """Get data loader"""
        try:
            if hasattr(self.datamodule, 'val_set'):
                from torch.utils.data import DataLoader
                return DataLoader(
                    dataset=self.datamodule.val_set,
                    batch_size=1,  # Single sample processing for visualization
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
                raise AttributeError("No available dataset found in data module")
        except Exception as e:
            log.error(f"Failed to create data loader: {e}")
            raise

    def generate_expert_gradcam(self, max_samples=5, save_dir='./gradcam_results'):
        """
        Generate expert Grad-CAM visualizations
        
        Args:
            max_samples: Maximum number of samples
            save_dir: Save directory
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        dataloader = self._get_dataloader()
        
        log.info("Starting MOE expert Grad-CAM visualization generation...")
        
        # Process each sample
        for batch_idx, batch_sample in enumerate(tqdm(dataloader, desc="Generating Grad-CAM")):
            if batch_idx >= max_samples:
                break
            
            # Generate expert Grad-CAM for current sample
            gradcam_results = self._generate_sample_gradcam(batch_sample, batch_idx)
            
            # Visualize and save results
            self._visualize_and_save_gradcam(gradcam_results, batch_idx, save_dir, batch_sample)
        
        log.info(f"Grad-CAM visualization completed, results saved in: {save_dir}")
    
    def _generate_sample_gradcam(self, batch_sample, sample_idx):
        """Generate expert Grad-CAM for a single sample"""
        batch_data = batch_sample['data'].to(self.device)
        batch_target = {key: value for key, value in batch_sample.items() if 'data' not in key}
        
        # Clear previous features and gradients
        self.expert_features.clear()
        self.expert_gradients.clear()
        
        # Register forward and backward hooks
        hooks = self._register_hooks()
        
        try:
            # Set model to training mode for gradient computation
            self.model.train()
            
            # Forward pass
            batch_data.requires_grad_(True)
            
            if hasattr(self.model, 'standardize'):
                batch_data_norm = self.model.standardize(batch_data)
            else:
                batch_data_norm = batch_data
            
            # Get model output
            output = self.model.net(batch_data_norm)
            
            # Compute Grad-CAM for each expert
            expert_gradcams = {}
            
            for expert_name in self.expert_names:
                if expert_name in self.expert_features:
                    # Compute target class loss (using maximum output value)
                    if isinstance(output, dict):
                        # For ACCDOA and other dictionary outputs
                        target_output = output.get('accdoa', output.get('sed', list(output.values())[0]))
                    else:
                        target_output = output
                    
                    # Use maximum output value as target
                    target_score = target_output.max()
                    
                    # Backward pass
                    self.model.zero_grad()
                    if target_score.requires_grad:
                        target_score.backward(retain_graph=True)
                    
                    # Compute Grad-CAM
                    gradcam_map = self._compute_gradcam(expert_name)
                    if gradcam_map is not None:
                        expert_gradcams[expert_name] = gradcam_map
            
            # Generate original spectrogram
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
            # Clean up hooks
            for hook in hooks:
                hook.remove()
            self.model.eval()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        hooks = []
        
        def forward_hook_fn(expert_name):
            def hook(module, input, output):
                # Store expert output features
                if isinstance(output, torch.Tensor):
                    self.expert_features[expert_name] = output.clone()
            return hook
        
        def backward_hook_fn(expert_name):
            def hook(module, grad_input, grad_output):
                # Store expert gradients
                if grad_output[0] is not None:
                    self.expert_gradients[expert_name] = grad_output[0].clone()
            return hook
        
        # Register hooks for each expert
        for expert_name, expert_module in self.expert_modules.items():
            # Forward hook
            forward_hook = expert_module.register_forward_hook(forward_hook_fn(expert_name))
            hooks.append(forward_hook)
            
            # Backward hook
            backward_hook = expert_module.register_backward_hook(backward_hook_fn(expert_name))
            hooks.append(backward_hook)
        
        return hooks
    
    def _compute_gradcam(self, expert_name):
        """Compute Grad-CAM for a single expert"""
        if expert_name not in self.expert_features or expert_name not in self.expert_gradients:
            log.warning(f"Expert {expert_name} missing feature or gradient information")
            return None
        
        features = self.expert_features[expert_name]  # [B, N, C] or [B, C, H, W]
        gradients = self.expert_gradients[expert_name]  # Corresponding gradients
        
        # Compute weights (global average pooling of gradients)
        if gradients.dim() == 3:  # [B, N, C] - Transformer features
            weights = torch.mean(gradients, dim=(0, 1))  # [C]
            
            # Weighted features
            weighted_features = features * weights.unsqueeze(0).unsqueeze(0)  # [B, N, C]
            
            # Sum along channel dimension
            gradcam = torch.sum(weighted_features, dim=2)  # [B, N]
            
            # Take first batch only
            gradcam = gradcam[0]  # [N]
            
            # Reshape sequence form gradcam to 2D
            gradcam = self._reshape_sequence_to_2d(gradcam)
            
        elif gradients.dim() == 4:  # [B, C, H, W] - CNN features
            weights = torch.mean(gradients, dim=(0, 2, 3))  # [C]
            
            # Weighted features
            weighted_features = features * weights.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [B, C, H, W]
            
            # Sum along channel dimension
            gradcam = torch.sum(weighted_features, dim=1)  # [B, H, W]
            gradcam = gradcam[0]  # [H, W]
        
        else:
            log.warning(f"Unsupported feature dimension: {gradients.dim()}")
            return None
        
        # ReLU activation (keep positive values only)
        gradcam = F.relu(gradcam)
        
        # Normalize to [0, 1]
        if gradcam.max() > gradcam.min():
            gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())
        
        return gradcam.detach().cpu().numpy()
    
    def _reshape_sequence_to_2d(self, sequence_tensor):
        """Reshape sequence form tensor to 2D"""
        N = sequence_tensor.shape[0]
        H = W = int(np.sqrt(N))
        
        if H * W != N:
            # If not a perfect square, use closest rectangle
            H = int(np.sqrt(N))
            W = N // H
            if H * W < N:
                H += 1
            # Pad to correct size
            if H * W > N:
                padding = H * W - N
                sequence_tensor = F.pad(sequence_tensor, (0, padding))
        
        return sequence_tensor.view(H, W)
    
    def _generate_mel_spectrogram(self, audio_data):
        """Generate Mel spectrogram"""
        # Get configuration parameters
        mel_params = self.cfg.visual.gradcam_analysis.mel_params
        sr = mel_params.sr
        n_mels = mel_params.n_mels
        n_fft = mel_params.n_fft
        hop_length = mel_params.hop_length
        fmax = mel_params.fmax
        
        # If multi-channel, take average
        if audio_data.ndim > 1:
            audio = np.mean(audio_data, axis=0)
        else:
            audio = audio_data
        
        # Compute Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_mels=n_mels, 
            n_fft=n_fft, 
            hop_length=hop_length,
            fmax=fmax
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel_spec
    
    def _visualize_and_save_gradcam(self, gradcam_results, sample_idx, save_dir, batch_sample):
        """Visualize and save Grad-CAM results - ALL ENGLISH TEXT"""
        original_spec = gradcam_results['original_spectrogram']
        expert_gradcams = gradcam_results['expert_gradcams']
        filename = gradcam_results['filename']
        
        # Get visualization parameters
        viz_params = self.cfg.visual.gradcam_analysis.visualization_params
        
        # Create multi-subplot visualization
        n_experts = len(expert_gradcams)
        if n_experts == 0:
            log.warning(f"Sample {sample_idx} has no valid Grad-CAM results")
            return
            
        fig, axes = plt.subplots(2, n_experts + 1, figsize=(4 * (n_experts + 1), 8))
        
        # Ensure axes is 2D array
        if n_experts == 0:
            return
        if axes.ndim == 1:
            axes = axes.reshape(1, -1)
        
        # First row: Original spectrogram + each expert's Grad-CAM heatmap
        # Original spectrogram
        im0 = axes[0, 0].imshow(original_spec, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 0].set_title('Original Log-Mel Spectrogram', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Time Frames')
        axes[0, 0].set_ylabel('Mel Frequency')
        plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)
        
        # Each expert's Grad-CAM
        for i, (expert_name, gradcam) in enumerate(expert_gradcams.items()):
            ax = axes[0, i + 1]
            im = ax.imshow(gradcam, aspect='auto', origin='lower', cmap='jet', alpha=0.8)
            ax.set_title(f'{expert_name}\nGrad-CAM Heatmap', fontsize=10, fontweight='bold')
            ax.set_xlabel('Time Frames')
            ax.set_ylabel('Frequency')
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Second row: Overlay visualization (Original spectrogram + Grad-CAM heatmap)
        # Original spectrogram (repeat display)
        axes[1, 0].imshow(original_spec, aspect='auto', origin='lower', cmap='viridis')
        axes[1, 0].set_title('Reference: Original Spectrogram', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Time Frames')
        axes[1, 0].set_ylabel('Mel Frequency')
        
        # Overlay visualization
        for i, (expert_name, gradcam) in enumerate(expert_gradcams.items()):
            ax = axes[1, i + 1]
            
            # First display original spectrogram as background
            ax.imshow(original_spec, aspect='auto', origin='lower', cmap='viridis', alpha=0.7)
            
            # Adjust gradcam size to match spectrogram
            if gradcam.shape != original_spec.shape:
                gradcam_resized = cv2.resize(gradcam, (original_spec.shape[1], original_spec.shape[0]))
            else:
                gradcam_resized = gradcam
            
            # Overlay Grad-CAM heatmap
            im_overlay = ax.imshow(gradcam_resized, aspect='auto', origin='lower', 
                                 cmap='hot', alpha=viz_params.overlay_alpha, vmin=0, vmax=1)
            
            ax.set_title(f'{expert_name}\nOverlay Visualization', fontsize=10, fontweight='bold')
            ax.set_xlabel('Time Frames')
            ax.set_ylabel('Frequency')
            plt.colorbar(im_overlay, ax=ax, shrink=0.8, label='Attention Intensity')
        
        # Overall title
        fig.suptitle(f'Sample {sample_idx}: {filename}\nMOE Expert Grad-CAM Visualization', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save image
        save_path = save_dir / f'gradcam_sample_{sample_idx}.png'
        plt.savefig(save_path, dpi=viz_params.dpi, bbox_inches='tight')
        plt.close()
        
        log.info(f"Sample {sample_idx} Grad-CAM visualization saved: {save_path}")
        
        # Save numerical results
        results_path = save_dir / f'gradcam_data_sample_{sample_idx}.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(gradcam_results, f)
    
    def generate_expert_comparison_heatmap(self, max_samples=10, save_path='./expert_comparison_heatmap.png'):
        """Generate expert comparison heatmap - ALL ENGLISH TEXT"""
        log.info("Generating expert comparison heatmap...")
        
        dataloader = self._get_dataloader()
        
        # Collect expert attention data from all samples
        all_expert_attention = {name: [] for name in self.expert_names}
        
        for batch_idx, batch_sample in enumerate(tqdm(dataloader, desc="Collecting expert attention data")):
            if batch_idx >= max_samples:
                break
            
            gradcam_results = self._generate_sample_gradcam(batch_sample, batch_idx)
            
            for expert_name, gradcam in gradcam_results['expert_gradcams'].items():
                # Calculate average attention intensity
                avg_attention = np.mean(gradcam)
                all_expert_attention[expert_name].append(avg_attention)
        
        # Create comparison heatmap
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
        
        log.info(f"Expert comparison heatmap saved: {save_path}")


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
            model.load_state_dict(checkpoint['state_dict'], strict=True)
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