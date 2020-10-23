# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Zigang Geng (aa397601@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import pycocotools
from .COCODataset import CocoDataset
from .target_generators import HeatmapGenerator


logger = logging.getLogger(__name__)


class CocoKeypoints(CocoDataset):
    def __init__(self,
                 cfg,
                 dataset_name,
                 remove_images_without_annotations,
                 heatmap_generator,
                 offset_generator=None,
                 transforms=None):
        super().__init__(cfg.DATASET.ROOT,
                         dataset_name,
                         cfg.DATASET.DATA_FORMAT,
                         cfg.DATASET.NUM_JOINTS,
                         cfg.DATASET.GET_RESCORE_DATA)  # 调用父类初始化方法

        if cfg.DATASET.WITH_CENTER:
            assert cfg.DATASET.NUM_JOINTS == 18, 'Number of joint with center for COCO is 18'
        else:
            assert cfg.DATASET.NUM_JOINTS == 17, 'Number of joint for COCO is 17'

        self.num_scales = self._init_check(heatmap_generator)
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.with_center = cfg.DATASET.WITH_CENTER
        self.num_joints_without_center = self.num_joints - 1 \
            if self.with_center else self.num_joints
        self.base_sigma = cfg.DATASET.BASE_SIGMA
        self.base_size = cfg.DATASET.BASE_SIZE
        self.min_sigma = cfg.DATASET.MIN_SIGMA
        self.center_sigma = cfg.DATASET.CENTER_SIGMA
        self.sigma = cfg.DATASET.SIGMA
        self.bg_weight = cfg.DATASET.BG_WEIGHT

        self.use_mask = cfg.DATASET.USE_MASK
        self.use_bbox_center = cfg.DATASET.USE_BBOX_CENTER

        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]

        self.transforms = transforms
        self.heatmap_generator = heatmap_generator
        self.offset_generator = offset_generator

    def __getitem__(self, idx):
        img, anno = super().__getitem__(idx)  # 调用父类键值关联方法

        mask = self.get_mask(anno, idx)  # 返回分割mask
        # TODO 为什么有的检测有的分割

        anno = [
            obj for obj in anno
            if obj['iscrowd'] == 0 or obj['num_keypoints'] > 0
        ]

        joints, area = self.get_joints(anno)  # 返回处理后的关节点注释和目标检测区域大小

        mask_list = [mask.copy() for _ in range(self.num_scales)]
        joints_list = [joints.copy() for _ in range(self.num_scales)]
        target_list = list()
        ind_mask_list = list()
        offset_list = list()
        weights_list = list()

        if self.transforms:
            img, mask_list, joints_list, area = self.transforms(
                img, mask_list, joints_list, area
            )  # 对当前图像进行变换

        # 根据尺度数目进行生成
        for scale_id in range(self.num_scales):
            scaled_target = []
            scaled_mask = []
            mask = mask_list[scale_id].copy()

            for i, sgm in enumerate(self.sigma[scale_id]):
                target_t, ignored_t = self.heatmap_generator[scale_id](
                    joints_list[scale_id],
                    sgm,
                    self.center_sigma,
                    self.bg_weight[scale_id][i])  # 生成当前尺度的热图

                scaled_mask.append((mask*ignored_t).astype(np.float32))
                scaled_target.append(target_t.astype(np.float32))

            if self.offset_generator is not None:
                offset_t, weight_t = self.offset_generator[scale_id](joints_list[scale_id], area)
                offset_list.append([offset_t])
                weights_list.append([weight_t])  # 生成当前尺度的偏移图

            target_list.append(scaled_target)
            ind_mask_list.append(scaled_mask)

        return img, target_list, ind_mask_list, offset_list, weights_list

    def get_joints(self, anno):
        num_people = len(anno)
        area = np.zeros((num_people, 1))
        joints = np.zeros((num_people, self.num_joints, 3))

        for i, obj in enumerate(anno):
            joints[i, :self.num_joints_without_center, :3] = \
                np.array(obj['keypoints']).reshape([-1, 3])  # 将一维列表转换为二维列表

            if self.use_mask == True:
                area[i, 0] = obj['area']
            else:
                area[i, 0] = obj['bbox'][2]*obj['bbox'][3]  # 目标检测的区域大小

            if self.with_center:
                if obj['area'] < 32**2:  # TODO 自定义数据集
                    joints[i, -1, 2] = 0  # 如果人体区域小于1024, 那么设置中心点为未标注
                    continue
                bbox = obj['bbox']
                center_x = (2*bbox[0] + bbox[2]) / 2.  # TODO 这里修改为上部分关节的中心位置与下部分关节的中心位置
                center_y = (2*bbox[1] + bbox[3]) / 2.  # 中心点位置为区域的中心
                joints_sum = np.sum(joints[i, :-1, :2], axis=0)  # sum前形状为(17,2), sum后形状为(2)
                num_vis_joints = len(np.nonzero(joints[i, :-1, 2])[0])  # 可视关节点的数量
                if self.use_bbox_center or num_vis_joints <= 0:
                    joints[i, -1, 0] = center_x
                    joints[i, -1, 1] = center_y
                else:
                    joints[i, -1, :2] = joints_sum / num_vis_joints  # 如果不使用盒的中心作为中心点, 就使用关节点的中心点
                    # TODO num_vis_joints1 = len(np.nonzero(joints[i, :10, 2])[0])
                    # joints_sum1 = np.sum(joints[i, :num_vis_joints1, :2], axis=0)
                    # joints_sum2 = np.sum(joints[i, num_vis_joints1+1:-1, :2], axis=0)
                    # joints[i, -1, :2] = joints_sum1 / num_vis_joints1
                    # joints[i, -2, :2] = joints_sum2 / num_vis_joints - num_vis_joints1
                    # TODO 试试关节点的中心点，或者指定中心点
                joints[i, -1, 2] = 1

        return joints, area

    def get_mask(self, anno, idx):
        coco = self.coco
        img_info = coco.loadImgs(self.ids[idx])[0]

        m = np.zeros((img_info['height'], img_info['width']))  # 生成零值mask

        # 生成多人分割mask
        for obj in anno:
            if obj['iscrowd']:
                rle = pycocotools.mask.frPyObjects(
                    obj['segmentation'], img_info['height'], img_info['width'])  # Encode RLE from a list of python objects
                m += pycocotools.mask.decode(rle)  # Decode binary masks encoded via RLE
            elif obj['num_keypoints'] == 0:
                rles = pycocotools.mask.frPyObjects(
                    obj['segmentation'], img_info['height'], img_info['width'])
                for rle in rles:
                    m += pycocotools.mask.decode(rle)

        return m < 0.5  # 返回背景mask

    def _init_check(self, heatmap_generator):
        assert isinstance(heatmap_generator, (list, tuple)
                          ), 'heatmap_generator should be a list or tuple'
        return len(heatmap_generator)
