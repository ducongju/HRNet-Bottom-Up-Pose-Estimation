# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tensorboardX import SummaryWriter

import _init_paths
import models

from config import cfg
from config import update_config
from core.loss import MultiLossFactory
from core.trainer import do_train
from dataset import make_dataloader
from utils.utils import create_logger
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import setup_logger


def parse_args():
    """对输入参数进行解析

    Returns: 解析实例

    """
    parser = argparse.ArgumentParser(description='Train keypoints network')  # 建立解析对象
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)  # 指定yaml文件的路径

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # distributed training
    parser.add_argument('--gpu',
                        help='gpu id for multiprocessing training',
                        type=str)
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')

    args = parser.parse_args()  # 构造解析实例

    return args


def main():
    args = parse_args()  # 对输入参数进行解析
    update_config(cfg, args)  # 根据输入参数对默认的配置进行更新

    # 更新解析实例中的主机rank值, 主节点为0
    cfg.defrost()
    cfg.RANK = args.rank
    cfg.freeze()

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train'
    )  # 创建logger

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or cfg.MULTIPROCESSING_DISTRIBUTED

    ngpus_per_node = torch.cuda.device_count()
    if cfg.MULTIPROCESSING_DISTRIBUTED:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, args, final_output_dir, tb_log_dir)
        )
    else:
        # Simply call main_worker function
        main_worker(
            ','.join([str(i) for i in cfg.GPUS]),
            ngpus_per_node,
            args,
            final_output_dir,
            tb_log_dir
        )


def main_worker(
        gpu, ngpus_per_node, args, final_output_dir, tb_log_dir
):
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if cfg.MULTIPROCESSING_DISTRIBUTED:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        print('Init process group: dist_url: {}, world_size: {}, rank: {}'.
              format(args.dist_url, args.world_size, args.rank))
        dist.init_process_group(
            backend=cfg.DIST_BACKEND,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )

    update_config(cfg, args)

    # setup logger
    logger, _ = setup_logger(final_output_dir, args.rank, 'train')

    # 根据配置文件构建网络结构
    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )

    # 拷贝lib/models/pose_hrnet.py文件到输出目录之中
    # copy model file
    if not cfg.MULTIPROCESSING_DISTRIBUTED or (
            cfg.MULTIPROCESSING_DISTRIBUTED
            and args.rank % ngpus_per_node == 0
    ):
        this_dir = os.path.dirname(__file__)
        shutil.copy2(
            os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
            final_output_dir
        )

    # 用于训练信息的图形化显示
    writer_dict = {
        'writer': SummaryWriter(logdir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            # args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    if not cfg.MULTIPROCESSING_DISTRIBUTED or (
            cfg.MULTIPROCESSING_DISTRIBUTED
            and args.rank % ngpus_per_node == 0
    ):
        #  用于模型的图形化显示
        dump_input = torch.rand(
            (1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE)
        ).cuda()
        # writer_dict['writer'].add_graph(model, (dump_input, ))
        logger.info(get_model_summary(model, dump_input, verbose=cfg.VERBOSE))

    # define loss function (criterion) and optimizer
    loss_factory = MultiLossFactory(cfg).cuda()

    # 创建训练数据的迭代器
    # Data loading code
    train_loader = make_dataloader(
        cfg, is_train=True, distributed=args.distributed
    )
    logger.info(train_loader.dataset)

    # 模型加载以及优化策略的相关配置
    best_perf = -1
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)

    # 自动从checkpoint中恢复训练
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth.tar')
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file, 
                        map_location=lambda storage, loc: storage)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    # 调整学习率
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    # 循环迭代进行训练
    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        lr_scheduler.step()

        # 根据给定的参数进行一个epoch的训练
        # train one epoch
        do_train(cfg, model, train_loader, loss_factory, optimizer, epoch,
                 final_output_dir, tb_log_dir, writer_dict)

        # 更新当前epoch的模型参数
        perf_indicator = epoch
        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        # 保存checkpoint和model_best文件,
        # 其中model_best是一个包括所有模块状态的字典, checkpoint是除此之外还包括epoch, model_name等参数
        if not cfg.MULTIPROCESSING_DISTRIBUTED or (
                cfg.MULTIPROCESSING_DISTRIBUTED
                and args.rank == 0
        ):
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            #best_model = True
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state{}.pth.tar'.format(gpu)
    )

    # 将两个CPU的训练结果保存至final_state{}.pth.tar
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
