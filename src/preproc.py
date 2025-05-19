import hydra
from omegaconf import DictConfig
from utils.config import get_dataset
from preproc.preprocess import Preprocess


@hydra.main(version_base="1.3", config_path="../configs", config_name="preproc.yaml")
def main(cfg: DictConfig):
    """数据预处理主函数
    
    根据配置执行数据预处理流程，包括数据提取和标签生成。
    
    参数:
        cfg (DictConfig): 由Hydra组成的配置对象，包含预处理参数
    """
    # 初始化数据集
    dataset = get_dataset(dataset_name=cfg.dataset, cfg=cfg)
    # 创建预处理器实例
    preprocessor = Preprocess(cfg, dataset)
    
    if cfg.mode == 'extract_data':
        # 提取数据和索引
        preprocessor.extract_index()
        
        # 特殊处理L3DAS22数据集
        if cfg.dataset == 'L3DAS22':
            preprocessor.extract_l3das22_label()

        # 对于STARSS23数据集的评估集，不处理标签（因为评估集没有标签）
        if cfg.dataset_type == 'eval' and cfg.dataset == 'STARSS23':
            return
            
        # 提取各种格式的标签
        preprocessor.extract_accdoa_label()  # 提取ACCDOA格式标签
        preprocessor.extract_track_label()   # 提取轨迹标签
        preprocessor.extract_adpit_label()   # 提取ADPIT格式标签


if __name__ == "__main__":
    main()
