# src/visual/visual.py

from typing import List
import hydra
import lightning as L
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from pathlib import Path
import logging
from tqdm import tqdm
import pickle
import sys
from omegaconf import DictConfig

# 修复路径问题 - 添加src目录到Python路径
current_dir = Path(__file__).parent
src_dir = current_dir.parent
project_root = src_dir.parent

# 添加src目录到路径，这样可以直接导入src下的模块
sys.path.insert(0, str(src_dir))

from utils.config import get_dataset
from utils.utilities import extras, get_pylogger

# 使用默认英文字体，避免中文字体兼容性问题
# plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False

log = get_pylogger(__name__)

class AdapterFeatureVisualizer:
    """适配器特征可视化工具类"""
    
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
        
        # 专家名称列表 (从配置文件中获取)
        self.expert_names = self._get_expert_names()
        
    def _get_expert_names(self):
        """从配置中获取专家名称"""
        experts_config = self.cfg.adapt.adapt_kwargs.get('experts_config', [])
        names = []
        for expert in experts_config:
            names.append(expert.get('name', f'expert_{len(names)}'))
        return names
    
    def extract_features_from_adapters(self, max_batches=10, layer_name='mlp'):
        """
        提取不同adapter的特征
        
        Args:
            max_batches: 最多处理的批次数
            layer_name: 要提取特征的层名称 ('mlp' 或 'attention')
            
        Returns:
            features_dict: {expert_name: features} 字典
            labels: 对应的样本标签
        """
        log.info(f"开始提取 {layer_name} 层的adapter特征...")
        
        features_dict = {name: [] for name in self.expert_names}
        labels = []
        
        # 获取数据加载器，避免使用trainer
        try:
            if hasattr(self.datamodule, 'val_set'):
                # 直接创建数据加载器，避免使用trainer
                from torch.utils.data import DataLoader
                dataloader = DataLoader(
                    dataset=self.datamodule.val_set,
                    batch_size=self.cfg.model.batch_size,
                    shuffle=False,
                    num_workers=self.cfg.num_workers,
                    pin_memory=True
                )
            elif hasattr(self.datamodule, 'test_set'):
                # 如果没有验证集，使用测试集
                from torch.utils.data import DataLoader
                dataloader = DataLoader(
                    dataset=self.datamodule.test_set,
                    batch_size=self.cfg.model.batch_size,
                    shuffle=False,
                    num_workers=self.cfg.num_workers,
                    pin_memory=True
                )
            else:
                raise AttributeError("无法找到可用的数据集")
        except Exception as e:
            log.error(f"创建数据加载器失败: {e}")
            return features_dict, labels
        
        # 注册钩子函数来收集特征
        hooks = []
        collected_features = {}
        
        def create_hook(expert_name):
            def hook_fn(module, input, output):
                if expert_name not in collected_features:
                    collected_features[expert_name] = []
                # 收集输出特征
                if isinstance(output, torch.Tensor):
                    collected_features[expert_name].append(output.detach().cpu())
            return hook_fn
        
        # 为每个专家注册钩子
        def register_hooks_recursive(module, prefix=""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                # 查找混合适配器模块
                if hasattr(child, 'experts') and hasattr(child, 'expert_names'):
                    log.info(f"找到混合适配器模块: {full_name}")
                    for i, expert in enumerate(child.experts):
                        expert_name = child.expert_names[i] if i < len(child.expert_names) else f"expert_{i}"
                        hook = expert.register_forward_hook(create_hook(expert_name))
                        hooks.append(hook)
                        log.info(f"为专家 {expert_name} 注册了钩子")
                
                register_hooks_recursive(child, full_name)
        
        # 注册所有钩子
        register_hooks_recursive(self.model.net)
        
        # 设置模型为评估模式
        self.model.eval()
        
        try:
            with torch.no_grad():
                for batch_idx, batch_sample in enumerate(tqdm(dataloader, desc="提取特征")):
                    if batch_idx >= max_batches:
                        break
                    
                    batch_data = batch_sample['data'].to(self.device)
                    batch_target = {key: value for key, value in batch_sample.items() if 'data' not in key}
                    
                    # 前向传播
                    try:
                        _ = self.model.common_step(batch_data, batch_target)
                    except Exception as e:
                        log.warning(f"common_step出错: {e}")
                        # 直接调用模型网络
                        try:
                            _ = self.model.net(batch_data)
                        except Exception as e2:
                            log.warning(f"直接调用net也出错: {e2}")
                            # 尝试标准化后再调用
                            batch_data_norm = self.model.standardize(batch_data)
                            _ = self.model.net(batch_data_norm)
                    
                    # 收集这个批次的特征
                    for expert_name in self.expert_names:
                        if expert_name in collected_features and collected_features[expert_name]:
                            # 取最新的特征（当前批次）
                            latest_features = collected_features[expert_name][-1]
                            # 对特征做全局平均池化以降维
                            if latest_features.dim() > 2:
                                latest_features = latest_features.mean(dim=1)  # [B, C]
                            features_dict[expert_name].append(latest_features)
                    
                    # 收集标签（可选）
                    for label_key in ['adpit_label', 'accdoa_label', 'sed_label']:
                        if label_key in batch_target:
                            labels.append(batch_target[label_key].cpu())
                            break
                    
        finally:
            # 移除所有钩子
            for hook in hooks:
                hook.remove()
        
        # 合并特征
        for expert_name in self.expert_names:
            if features_dict[expert_name]:
                features_dict[expert_name] = torch.cat(features_dict[expert_name], dim=0).numpy()
            else:
                log.warning(f"专家 {expert_name} 没有收集到特征")
                
        log.info("特征提取完成")
        return features_dict, labels
    
    def visualize_tsne(self, features_dict, save_path=None, title='t-SNE Visualization of Different Adapters', perplexity=30):
        """
        使用t-SNE进行特征可视化
        
        Args:
            features_dict: 特征字典
            save_path: 保存路径
            title: 图标题
            perplexity: t-SNE参数
        """
        log.info("开始t-SNE可视化...")
        
        # 合并所有特征
        all_features = []
        all_labels = []
        
        for expert_name, features in features_dict.items():
            if len(features) > 0:
                all_features.append(features)
                all_labels.extend([expert_name] * len(features))
        
        if not all_features:
            log.error("没有找到有效的特征数据")
            return
            
        all_features = np.concatenate(all_features, axis=0)
        
        # 如果特征维度太高，先用PCA降维
        if all_features.shape[1] > 50:
            log.info(f"特征维度过高 ({all_features.shape[1]})，先使用PCA降维到50维")
            pca = PCA(n_components=50)
            all_features = pca.fit_transform(all_features)
        
        # t-SNE降维
        log.info("执行t-SNE降维...")
        perplexity = min(perplexity, len(all_features)//4)
        if perplexity < 5:
            perplexity = 5
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        features_2d = tsne.fit_transform(all_features)
        
        # 绘图
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.expert_names)))
        
        for i, expert_name in enumerate(self.expert_names):
            if expert_name in features_dict and len(features_dict[expert_name]) > 0:
                mask = np.array(all_labels) == expert_name
                if np.any(mask):
                    plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                              c=[colors[i]], label=expert_name, alpha=0.6, s=50)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(title, fontsize=14)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            log.info(f"图像已保存到: {save_path}")
        
        plt.show()
        
    def visualize_feature_statistics(self, features_dict, save_path=None):
        """
        可视化特征统计信息
        
        Args:
            features_dict: 特征字典
            save_path: 保存路径
        """
        log.info("生成特征统计可视化...")
        
        stats = {}
        for expert_name, features in features_dict.items():
            if len(features) > 0:
                stats[expert_name] = {
                    'Mean': np.mean(features),
                    'Std': np.std(features),
                    'Max': np.max(features),
                    'Min': np.min(features)
                }
        
        if not stats:
            log.error("没有找到有效的特征数据")
            return
        
        # 绘制统计图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        stat_names = ['Mean', 'Std', 'Max', 'Min']
        
        for i, stat_name in enumerate(stat_names):
            expert_names = list(stats.keys())
            values = [stats[name][stat_name] for name in expert_names]
            
            axes[i].bar(expert_names, values, alpha=0.7)
            axes[i].set_title(f'Feature {stat_name} Comparison', fontsize=12)
            axes[i].set_ylabel(stat_name, fontsize=10)
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            log.info(f"统计图已保存到: {save_path}")
            
        plt.show()
    
    def save_features(self, features_dict, save_path):
        """保存提取的特征"""
        with open(save_path, 'wb') as f:
            pickle.dump(features_dict, f)
        log.info(f"特征已保存到: {save_path}")
    
    def load_features(self, load_path):
        """加载保存的特征"""
        with open(load_path, 'rb') as f:
            features_dict = pickle.load(f)
        log.info(f"特征已从 {load_path} 加载")
        return features_dict


@hydra.main(version_base="1.3", config_path="../../configs", config_name="visual.yaml")
def main(cfg: DictConfig):
    """可视化主函数
    
    根据配置执行adapter特征可视化。
    
    参数:
        cfg (DictConfig): 由Hydra组成的配置对象，包含可视化参数
    """

    # 应用额外的工具函数
    extras(cfg)

    # 设置随机种子，确保实验可重复性
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
   
    # 确保使用验证模式
    if not hasattr(cfg, 'mode'):
        cfg.mode = 'valid'
    
    if cfg.mode == 'valid':
        # 验证模式
        # 获取默认验证数据集名称
        default_dataset = list(cfg.data.valid_dataset.keys())[0]
        # 初始化数据集
        dataset = get_dataset(dataset_name=default_dataset, cfg=cfg)

        # 实例化数据模块
        log.info(f"正在实例化数据模块 <{cfg.datamodule._target_}> ...")
        datamodule = hydra.utils.instantiate(cfg.datamodule, cfg, dataset, 'fit')
        # 获取验证集元数据
        valid_meta = datamodule.paths_dict, datamodule.valid_gt_dcaseformat

        # 实例化模型
        log.info(f"正在实例化模型 <{cfg.modelmodule._target_}> ...")
        model = hydra.utils.instantiate(cfg.modelmodule, cfg, dataset, valid_meta)
        
        # 设置模型（初始化网络）
        log.info("正在设置模型...")
        model.setup('valid')

    elif cfg.mode == 'test':
        # 测试模式
        # 获取默认测试数据集名称
        default_dataset = list(cfg.data.test_dataset.keys())[0]
        # 初始化数据集
        dataset = get_dataset(dataset_name=default_dataset, cfg=cfg)

        # 实例化数据模块
        log.info(f"正在实例化数据模块 <{cfg.datamodule._target_}> ...")
        datamodule = hydra.utils.instantiate(cfg.datamodule, cfg, dataset, 'test')
        # 获取测试集元数据
        test_meta = datamodule.paths_dict

        # 实例化模型
        log.info(f"正在实例化模型 <{cfg.modelmodule._target_}> ...")
        model = hydra.utils.instantiate(cfg.modelmodule, cfg, dataset, test_meta=test_meta)
        
        # 设置模型（初始化网络）
        log.info("正在设置模型...")
        model.setup('test')

    # 加载模型权重
    if cfg.get("ckpt_path"):
        log.info(f"正在加载模型权重: {cfg.ckpt_path}")
        checkpoint = torch.load(cfg.ckpt_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    # 移动模型到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 创建可视化器
    log.info("正在创建特征可视化器...")
    visualizer = AdapterFeatureVisualizer(cfg, model, datamodule)
    
    # 开始可视化
    log.info("开始特征提取和可视化!")
    
    # 提取特征
    max_batches = cfg.get('visual', {}).get('max_batches', 10)
    features_dict, labels = visualizer.extract_features_from_adapters(max_batches=max_batches)
    
    # 保存特征
    all_save_path = Path(cfg.paths.output_dir)

    features_save_path = "adapter_features.pkl"
    visualizer.save_features(features_dict, all_save_path / features_save_path)
    
    # t-SNE可视化
    tsne_save_path = "adapter_tsne_visualization.png"
    visualizer.visualize_tsne(
        features_dict, 
        save_path=all_save_path / tsne_save_path,
        title="t-SNE Visualization of Different Adapters"
    )
    
    # 统计信息可视化
    stats_save_path = "adapter_statistics.png"
    visualizer.visualize_feature_statistics(
        features_dict,
        save_path=all_save_path / stats_save_path
    )
    
    log.info("可视化完成！")
    log.info(f"特征数据保存在: {features_save_path}")
    log.info(f"t-SNE图保存在: {tsne_save_path}")
    log.info(f"统计图保存在: {stats_save_path}")


if __name__ == "__main__":
    # 设置矩阵乘法的精度为中等，在性能和精度之间取得平衡
    torch.set_float32_matmul_precision('medium')
    
    # 执行主函数
    main()