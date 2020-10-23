# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data

from .COCODataset import CocoDataset as coco
from .COCODatasetGetScoreData import CocoDatasetGetScoreData as cocoscore
from .CrowdPoseDatasetGetScoreData import CrowdPoseDatasetGetScoreData as crowdposescore
from .COCOKeypoints import CocoKeypoints as coco_kpt
from .CrowdPoseDataset import CrowdPoseDataset as crowd_pose
from .CrowdPoseKeypoints import CrowdPoseKeypoints as crowd_pose_kpt
from .transforms import build_transforms
from .target_generators import HeatmapGenerator
from .target_generators import OffsetGenerator


def build_dataset(cfg, is_train):
    """创建数据集对象

    Args:
        cfg (yaml): 配置文件
        is_train (bool): 该数据集是否用于训练

    Returns:
        dataset (dataset): 处理后的数据集

    """
    transforms = build_transforms(cfg, is_train)  # 创建图像变换对象, 详见内部

    _HeatmapGenerator = HeatmapGenerator  # TODO 为什么要转换成私有类

    heatmap_generator = [
        _HeatmapGenerator(
            output_size, cfg.DATASET.NUM_JOINTS
        ) for output_size in cfg.DATASET.OUTPUT_SIZE
    ]  # 创建热图生成对象, 详见内部

    offset_generator = None
    if cfg.DATASET.OFFSET_REG:
        offset_generator = [
            OffsetGenerator(
                output_size,
                output_size,
                cfg.DATASET.NUM_JOINTS,
                cfg.DATASET.OFFSET_RADIUS
            ) for output_size in cfg.DATASET.OUTPUT_SIZE
        ]  # 创建偏移生成对象, 详见内部

    dataset_name = cfg.DATASET.TRAIN if is_train else cfg.DATASET.TEST

    dataset = eval(cfg.DATASET.DATASET)(
        cfg,
        dataset_name,
        is_train,
        heatmap_generator,
        offset_generator,
        transforms
    )  # 选择dataset对象是由CocoKeypoints类创建还是CrowdPoseKeypoints类创建, 详见COCOKeypoints

    return dataset


def make_dataloader(cfg, is_train=True, distributed=False):
    """创建数据加载器对象

    Args:
        cfg (yaml): 配置文件
        is_train (bool): 该数据集是否用于训练
        distributed (bool):  是否实现分布式训练

    Returns:
        data_loader (object): 数据加载器

    """
    if is_train:
        images_per_gpu = cfg.TRAIN.IMAGES_PER_GPU
        shuffle = True
    else:
        images_per_gpu = cfg.TEST.IMAGES_PER_GPU
        shuffle = False
    images_per_batch = images_per_gpu * len(cfg.GPUS)  # 每批处理的图像等于每个GPU处理的图像数量乘以GPU数量

    dataset = build_dataset(cfg, is_train)  # 详见内部

    if is_train and distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset
        )  # 确保dataloader只会load到整个数据集的一个特定子集
        shuffle = False
    else:
        train_sampler = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=images_per_batch,
        shuffle=shuffle,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        sampler=train_sampler
    )

    return data_loader


def make_test_dataloader(cfg):
    """创建数据加载器对象

    Args:
        cfg (yaml): 配置文件

    Returns:
        data_loader (object): 数据加载器
        dataset (object): 处理后的数据集

    """
    transforms = None
    dataset = eval(cfg.DATASET.DATASET_TEST)(
        cfg.DATASET.ROOT,
        cfg.DATASET.TEST,
        cfg.DATASET.DATA_FORMAT,
        cfg.DATASET.NUM_JOINTS,
        cfg.DATASET.GET_RESCORE_DATA,
        transforms,
        bbox_file=cfg.TEST.BBOX_FILE if cfg.TEST.BBOX_GROUPING else None
    )  # 选择dataset对象是由CocoDataset类创建还是CrowdPoseDataset类创建, 详见COCODataset

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return data_loader, dataset
