# Reference: https://github.com/ashleve/lightning-hydra-template

# 测试脚本
from typing import List

import hydra
import lightning as L
import torch
from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from utils.config import get_dataset
from utils.utilities import (extras, get_pylogger, instantiate_callbacks,
                             instantiate_loggers, log_hyperparameters)

# 初始化日志记录器
log = get_pylogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """模型训练主函数
    
    使用Hydra管理配置，初始化并训练模型。
    
    参数:
        cfg (DictConfig): 由Hydra组成的配置对象，包含所有训练参数
    """

    # 应用额外的工具函数
    # (例如：如果没有提供标签，则请求输入；打印配置树等)
    extras(cfg)

    # 设置随机种子，确保实验可重复性
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
   
    # 获取默认数据集名称
    default_dataset = list(cfg.data.train_dataset.keys())[0]
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

    # 实例化回调函数
    log.info("正在实例化回调函数...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # 实例化日志记录器
    log.info("正在实例化日志记录器...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # 实例化训练器
    log.info(f"正在实例化训练器 <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    # 创建对象字典，用于记录超参数
    object_dict = {
        "cfg": cfg,                # 配置对象
        "datamodule": datamodule,  # 数据模块
        "model": model,            # 模型
        "callbacks": callbacks,    # 回调函数
        "logger": logger,          # 日志记录器
        "trainer": trainer,        # 训练器
    }

    # 如果存在日志记录器，记录超参数
    if logger:
        log.info("正在记录超参数!")
        log_hyperparameters(object_dict)
    
    # 开始训练
    log.info("开始训练!")

    # 执行模型训练
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    
    
if __name__ == "__main__":
    # 设置矩阵乘法的精度为中等，在性能和精度之间取得平衡
    torch.set_float32_matmul_precision('medium')
    
    # 执行主函数
    main()