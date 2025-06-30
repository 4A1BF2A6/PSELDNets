from pathlib import Path
import sys
import logging
import warnings
import pickle
from typing import List, Dict, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import hydra
import lightning as L
from omegaconf import DictConfig
from tqdm import tqdm
import librosa
import cv2
from sklearn.preprocessing import MinMaxScaler

# 修复路径问题 - 添加src目录到Python路径
current_dir = Path(__file__).parent
src_dir = current_dir.parent
project_root = src_dir.parent
# 添加src目录到路径，这样可以直接导入src下的模块
sys.path.insert(0, str(src_dir))
from utils.config import get_dataset
from utils.utilities import extras, get_pylogger

log = get_pylogger(__name__)

class ExpertActivationVisualizer:
    """专家激活概率可视化工具"""
    
    def __init__(self, cfg, model, datamodule):
        self.cfg = cfg
        self.model = model
        self.datamodule = datamodule
        self.device = next(model.parameters()).device
        
        # 从配置获取专家名称
        self.expert_names = self._get_expert_names()
        
    def _get_expert_names(self):
        """从配置中获取专家名称"""
        try:
            experts_config = self.cfg.adapt.adapt_kwargs.get('experts_config', [])
            names = []
            for expert in experts_config:
                names.append(expert.get('name', f'expert_{len(names)}'))
            return names
        except:
            # 默认专家名称
            return ['dct_expert', 'SE_expert', 'base_expert_1', 'mona_expert']
    
    def _get_dataloader(self):
        """获取数据加载器"""
        if hasattr(self.datamodule, 'val_set'):
            from torch.utils.data import DataLoader
            return DataLoader(
                dataset=self.datamodule.val_set,
                batch_size=self.cfg.model.batch_size,
                shuffle=False,
                num_workers=self.cfg.get('num_workers', 0),
                pin_memory=True
            )
        else:
            raise AttributeError("无法找到验证数据集") 

    def collect_block_expert_probs(self, max_batches=10):
        # 获取数据加载器
        dataloader = self._get_dataloader()
        # 获取所有block的mlp.adapter_instance.router模块列表
        routers = get_block_mlp_routers(self.model)
        n_blocks = len(routers)  # block数量
        n_experts = len(self.expert_names)  # 专家数量
        # 用于收集每个block的所有batch的专家激活概率
        collected_probs = [ [] for _ in range(n_blocks) ]

        # 定义hook工厂函数，每个block注册一个hook
        def make_hook(block_idx):
            def hook(module, input, output):
                # 兼容不同输出格式，通常output为[batch, n_experts]
                if isinstance(output, tuple):
                    router_scores = output[0]
                else:
                    router_scores = output
                # 如果输出维度大于2，拉平成[batch, n_experts]
                if router_scores.dim() > 2:
                    router_scores = router_scores.view(-1, router_scores.shape[-1])
                # 对每个样本做softmax，得到专家激活概率
                probs = torch.softmax(router_scores, dim=-1)
                # 收集当前batch的概率，转为numpy后存入对应block
                collected_probs[block_idx].append(probs.detach().cpu().numpy())
            return hook

        hooks = [router.register_forward_hook(make_hook(i)) for i, router in enumerate(routers)]

        self.model.eval()
        try:
            with torch.no_grad():
                for batch_idx, batch_sample in enumerate(tqdm(dataloader, desc="提取特征")):
                    if batch_idx >= max_batches:
                        break
                    batch_data = batch_sample['data'].to(self.device)
                    batch_target = {key: value for key, value in batch_sample.items() if 'data' not in key}
                    try:
                        _ = self.model.common_step(batch_data, batch_target)
                    except Exception as e:
                        log.warning(f"common_step失败: {e}")
                        if hasattr(self.model, 'standardize'):
                            batch_data_norm = self.model.standardize(batch_data)
                        else:
                            batch_data_norm = batch_data
                        _ = self.model.net(batch_data_norm)
        finally:
            for hook in hooks:
                hook.remove()

        # 统计每个block每个专家的平均激活概率
        block_expert_probs = np.zeros((n_blocks, n_experts))
        for i in range(n_blocks):
            if collected_probs[i]:
                all_probs = np.concatenate(collected_probs[i], axis=0)  # [n_samples, n_experts]
                block_expert_probs[i] = np.mean(all_probs, axis=0)
        return block_expert_probs  # shape: [n_blocks, n_experts]

    def plot_block_expert_heatmap(self, block_expert_probs, save_path=None, title="Activation Probability of Experts in Each Block"):
        plt.figure(figsize=(10, 5))
        ax = plt.gca()

         # 专家名称作为y轴标签
        expert_labels = self.expert_names if len(self.expert_names) == block_expert_probs.shape[1] else [f'Expert {i+1}' for i in range(block_expert_probs.shape[1])]
        
        #归一化每一列（每个block内最大为1），突出每个block内最活跃的专家
        norm_probs = block_expert_probs / (block_expert_probs.max(axis=1, keepdims=True) + 1e-8)

        sns.heatmap(block_expert_probs.T, cmap='viridis', square=True, cbar=True,
                    linewidths=1, linecolor='white',
                    xticklabels=[str(i+1) for i in range(block_expert_probs.shape[0])],
                    # yticklabels=[str(i+1) for i in range(block_expert_probs.shape[1])]
                    yticklabels=expert_labels,
                    )

        plt.xlabel("Block Index")
        plt.ylabel("Expert Index")
        plt.title(title)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            log.info(f"专家-Block激活概率热力图已保存到: {save_path}")
        plt.show()

def get_block_mlp_routers(model):
    """
    返回所有block的mlp.adapter_instance.router模块列表
    """
    routers = []
    # 假设结构为 encoder.layers.{i}.blocks.{j}.mlp.adapter_instance.router
    encoder = model.net.encoder
    for layer in encoder.layers:
        for block in layer.blocks:
            if hasattr(block, 'mlp') and hasattr(block.mlp, 'adapter_instance'):
                adapter = block.mlp.adapter_instance
                if hasattr(adapter, 'router'):
                    routers.append(adapter.router)
    return routers


@hydra.main(version_base="1.3", config_path="../../configs", config_name="expert_activation_visual.yaml")
def main(cfg: DictConfig):
    """时频适配器可视化主函数"""
    
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
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)
    
    # 移动模型到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 保存结果
    save_dir = Path(cfg.paths.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 创建专家激活可视化器
    log.info("正在创建专家激活可视化器...")
    visualizer = ExpertActivationVisualizer(cfg, model, datamodule)
    block_expert_probs = visualizer.collect_block_expert_probs(max_batches=cfg.visual.num_samples)

    # 保存特征
    all_save_path = Path(cfg.paths.output_dir)

    visualizer.plot_block_expert_heatmap(block_expert_probs, save_path=all_save_path / "block_expert_probs.png")
    
if __name__ == "__main__":
    # 设置矩阵乘法精度
    torch.set_float32_matmul_precision('medium')
    
    # 执行主函数
    main() 