# Reference: https://github.com/ashleve/lightning-hydra-template

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


@hydra.main(version_base="1.3", config_path="../configs", config_name="infer.yaml")
def main(cfg: DictConfig):
    """模型推理主函数
    
    根据配置执行模型验证或测试。
    
    参数:
        cfg (DictConfig): 由Hydra组成的配置对象，包含推理参数
    """

    # 应用额外的工具函数
    # (例如：如果没有提供标签，则请求输入；打印配置树等)
    extras(cfg)

    # 设置随机种子，确保实验可重复性
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
   
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
    
    # 开始推理
    log.info("开始推理!")

    if cfg.mode == 'valid':
        # 执行模型验证
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        # 注释掉的代码展示了如何逐个验证数据集
        # valid_dataset = cfg.data.valid_dataset
        # for key, value in valid_dataset.items():
        #     cfg.data.valid_dataset = {key: value}
        #     print(cfg.data.valid_dataset)
        #     datamodule = hydra.utils.instantiate(cfg.datamodule, cfg, dataset, 'fit')
        #     valid_meta = datamodule.paths_dict, datamodule.valid_gt_dcaseformat
        #     model.valid_paths_dict, model.valid_gt_dcase_format = valid_meta
        #     trainer.validate(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    elif cfg.mode == 'test':
        # 执行模型测试
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    
    
if __name__ == "__main__":
    # 设置矩阵乘法的精度为中等，在性能和精度之间取得平衡
    torch.set_float32_matmul_precision('medium')
    
    # 执行主函数
    main()