# 导入必要的库
import argparse  # 用于解析命令行参数
import collections  # 提供特殊容器数据类型
import random  # 用于生成随机数
import sys  # 提供与Python解释器交互的变量和函数
from pathlib import Path  # 提供面向对象的文件系统路径

import numpy as np  # 用于科学计算的库
import PIL  # Python图像处理库
import timm  # 计算机视觉模型库
import torch  # PyTorch深度学习框架
import torchvision  # PyTorch的计算机视觉工具包
from prettytable import PrettyTable  # 用于创建格式化的表格输出

# 导入自定义模块
from domainbed import hparams_registry  # 超参数注册表
from domainbed.datasets import get_dataset  # 数据集加载函数
from domainbed.eval import eval_en  # 评估函数
from domainbed.lib import misc  # 杂项工具函数
from domainbed.lib.logger import Logger  # 日志记录器
from domainbed.lib.writers import get_writer  # 获取写入器
from domainbed.trainer import train  # 训练函数
from sconf import Config  # 配置管理


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Domain generalization", allow_abbrev=False)
    # 添加各种命令行参数
    parser.add_argument("name", type=str)  # 实验名称
    parser.add_argument("configs", nargs="*")  # 配置文件列表
    parser.add_argument("--data_dir", type=str, default="datadir/")  # 数据目录
    parser.add_argument("--dataset", type=str, default="PACS")  # 数据集名称
    parser.add_argument("--algorithm", type=str, default="ERM")  # 算法名称
    parser.add_argument(
        "--trial_seed",
        type=int,
        default=0,
        help="Trial number (used for seeding split_dataset and random_hparams).",  # 用于数据集分割和随机超参数的种子
    )
    parser.add_argument("--r", type=int, default=4, help="Rank of adapter.")  # 适配器的秩
    parser.add_argument("--seed", type=int, default=0, help="Seed for everything else")  # 其他随机性的种子
    parser.add_argument("--steps", type=int, default=None)  # 训练步数
    parser.add_argument("--attention", type=bool, default=None)  # 是否使用注意力机制
    parser.add_argument("--l_aux", action="store_true", help="Use auxiliary loss")  # 是否使用辅助损失
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=None,
        help="Checkpoint every N steps. Default is dataset-dependent.",  # 检查点保存频率
    )
    parser.add_argument("--test_envs", type=int, nargs="+", default=None)  # 测试环境
    parser.add_argument("--holdout_fraction", type=float, default=0.2)  # 保留集比例
    parser.add_argument("--model_save", default=None, type=int, help="Model save start step")  # 模型保存起始步数
    parser.add_argument("--deterministic", action="store_true")  # 是否使用确定性模式

    parser.add_argument("--tb_freq", default=10)  # TensorBoard记录频率
    parser.add_argument("--debug", action="store_true", help="Run w/ debug mode")  # 是否使用调试模式
    parser.add_argument("--show", action="store_true", help="Show args and hparams w/o run")  # 是否只显示参数而不运行
    parser.add_argument(
        "--evalmode",
        default="fast",
        help="[fast, all]. if fast, ignore train_in datasets in evaluation time.",  # 评估模式
    )
    parser.add_argument("--prebuild_loader", action="store_true", help="Pre-build eval loaders")  # 是否预构建评估数据加载器
    parser.add_argument("--en", type=bool, default=None)  # 是否使用集成学习
    args, left_argv = parser.parse_known_args()  # 解析命令行参数
    args.deterministic = True  # 设置确定性模式为True

    # 设置超参数
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)  # 获取默认超参数
    # hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,args.trial_seed)  # 获取随机超参数

    # 加载配置文件
    keys = ["config.yaml"] + args.configs  # 配置文件列表
    keys = [open(key, encoding="utf8") for key in keys]  # 打开配置文件
    hparams = Config(*keys, default=hparams)  # 创建配置对象
    hparams.argv_update(left_argv)  # 更新配置参数

    # 设置调试模式
    if args.debug:
        args.checkpoint_freq = 5  # 调试模式下的检查点频率
        args.steps = 10  # 调试模式下的训练步数
        args.name += "_debug"  # 添加调试标记

    timestamp = misc.timestamp()  # 获取时间戳
    args.unique_name = f"{timestamp}_{args.name}"  # 创建唯一名称

    # 设置路径
    args.work_dir = Path(".")  # 工作目录
    args.data_dir = Path(args.data_dir)  # 数据目录

    args.out_root = args.work_dir / Path("train_output") / args.dataset  # 输出根目录
    args.out_dir = args.out_root / args.unique_name  # 输出目录
    args.out_dir.mkdir(exist_ok=True, parents=True)  # 创建输出目录

    # 设置日志和写入器
    writer = get_writer(args.out_root / "runs" / args.unique_name)  # 获取写入器
    logger = Logger.get(args.out_dir / "log.txt")  # 获取日志记录器
    if args.debug:
        logger.setLevel("DEBUG")  # 设置调试日志级别
    cmd = " ".join(sys.argv)  # 获取完整命令行
    logger.info(f"Command :: {cmd}")  # 记录命令行

    # 记录环境信息
    logger.nofmt("Environment:")
    logger.nofmt("\tPython: {}".format(sys.version.split(" ")[0]))  # Python版本
    logger.nofmt("\tPyTorch: {}".format(torch.__version__))  # PyTorch版本
    logger.nofmt("\tTorchvision: {}".format(torchvision.__version__))  # Torchvision版本
    logger.nofmt("\tCUDA: {}".format(torch.version.cuda))  # CUDA版本
    logger.nofmt("\tCUDNN: {}".format(torch.backends.cudnn.version()))  # CUDNN版本
    logger.nofmt("\tNumPy: {}".format(np.__version__))  # NumPy版本
    logger.nofmt("\tPIL: {}".format(PIL.__version__))  # PIL版本
    logger.nofmt("\ttimm: {}".format(timm.__version__))  # timm版本

    # 检查CUDA可用性
    assert torch.cuda.is_available(), "CUDA is not available"  # 确保CUDA可用

    # 记录参数信息
    logger.nofmt("Args:")
    for k, v in sorted(vars(args).items()):
        logger.nofmt("\t{}: {}".format(k, v))  # 记录每个参数

    # 记录超参数信息
    logger.nofmt("HParams:")
    for line in hparams.dumps().split("\n"):
        logger.nofmt("\t" + line)  # 记录每个超参数

    if args.show:
        exit()  # 如果只显示参数，则退出

    # 设置随机种子
    random.seed(args.seed)  # 设置Python随机种子
    np.random.seed(args.seed)  # 设置NumPy随机种子
    torch.manual_seed(args.seed)  # 设置PyTorch随机种子
    torch.cuda.manual_seed(args.seed)  # 设置CUDA随机种子
    torch.cuda.manual_seed_all(args.seed)  # 设置所有CUDA设备的随机种子
    torch.backends.cudnn.deterministic = args.deterministic  # 设置CUDNN确定性模式
    torch.backends.cudnn.benchmark = not args.deterministic  # 设置CUDNN基准测试模式

    # 创建虚拟数据集用于记录信息
    dataset, _in_splits, _out_splits = get_dataset([0], args, hparams)  # 获取数据集

    # 打印数据集信息
    logger.nofmt("Dataset:")
    logger.nofmt(f"\t[{args.dataset}] #envs={len(dataset)}, #classes={dataset.num_classes}")  # 记录数据集环境数和类别数
    for i, env_property in enumerate(dataset.environments):
        logger.nofmt(f"\tenv{i}: {env_property} (#{len(dataset[i])})")  # 记录每个环境的信息
    logger.nofmt("")

    # 设置训练步数和检查点频率
    n_steps = args.steps or dataset.N_STEPS  # 获取训练步数
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ  # 获取检查点频率
    logger.info(f"n_steps = {n_steps}")  # 记录训练步数
    logger.info(f"checkpoint_freq = {checkpoint_freq}")  # 记录检查点频率

    # 调整训练步数以适应检查点
    org_n_steps = n_steps  # 保存原始步数
    n_steps = (n_steps // checkpoint_freq) * checkpoint_freq + 1  # 调整步数
    logger.info(f"n_steps is updated to {org_n_steps} => {n_steps} for checkpointing")  # 记录步数更新

    # 设置测试环境
    if not args.test_envs:
        args.test_envs = [[te] for te in range(len(dataset))]  # 如果没有指定测试环境，则使用所有环境
    logger.info(f"Target test envs = {args.test_envs}")  # 记录目标测试环境

    ###########################################################################
    # 运行训练
    ###########################################################################
    all_records = []  # 存储所有记录
    results = collections.defaultdict(list)  # 存储结果

    # 对每个测试环境进行训练
    for test_env in args.test_envs:
        # 执行训练
        res, records = train(
            test_env,
            args=args,
            hparams=hparams,
            n_steps=n_steps,
            checkpoint_freq=checkpoint_freq,
            logger=logger,
            writer=writer,
        )

        all_records.append(records)  # 添加记录
        for k, v in res.items():
            results[k].append(v)  # 添加结果

    # 记录总结表格
    logger.info("=== Summary ===")  # 记录总结标题
    logger.info(f"Command: {' '.join(sys.argv)}")  # 记录完整命令行
    logger.info("Unique name: %s" % args.unique_name)  # 记录唯一名称
    logger.info("Out path: %s" % args.out_dir)  # 记录输出路径
    logger.info("Algorithm: %s" % args.algorithm)  # 记录算法名称
    logger.info("Dataset: %s" % args.dataset)  # 记录数据集名称

    # 创建结果表格
    table = PrettyTable(["Selection"] + dataset.environments + ["Avg."])  # 创建表格
    for key, row in results.items():
        row.append(np.mean(row))  # 计算平均值
        row = [f"{acc:.3%}" for acc in row]  # 格式化准确率
        table.add_row([key] + row)  # 添加行
    logger.nofmt(table)  # 记录表格


if __name__ == "__main__":
    main()  # 运行主函数
