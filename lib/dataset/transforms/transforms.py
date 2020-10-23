# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        """创建变换操作的对象的组合

        Args:
            transforms (object): 变换操作

        """
        self.transforms = transforms

    def __call__(self, image, mask, joints, area):
        """对输入image, mask, joints, area进行处理

        Args:
            image (ndarray: img_h, img_w, 3): 原始图像
            mask (list: length(output_size)=2) (ndarray: img_h, img_w): 人体分割的掩膜
            joints (list: length(output_size)=2) (ndarray: num_people, num_joints=17, visible=3): 关节点位置和是否可视
            area (ndarray: num_people, 1): 人体检测框的面积

        Returns: 处理后的image, mask, joints, area

        """
        for t in self.transforms:
            image, mask, joints, area = t(image, mask, joints, area)
        return image, mask, joints, area

    # 显示实例的属性
    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    # Convert a PIL Image or numpy.ndarray to tensor
    def __call__(self, image, mask, joints, area):
        return F.to_tensor(image), mask, joints, area


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    # 返回归一化图像
    def __call__(self, image, mask, joints, area):
        # output[channel] = (input[channel] - mean[channel]) / std[channel]
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, mask, joints, area


class RandomHorizontalFlip(object):
    def __init__(self, flip_index, output_size, prob=0.5):
        self.flip_index = flip_index
        self.prob = prob
        self.output_size = output_size if isinstance(output_size, list) \
            else [output_size]

    # 返回图像的随机翻转
    def __call__(self, image, mask, joints, area):
        assert isinstance(mask, list)
        assert isinstance(joints, list)
        assert len(mask) == len(joints)
        assert len(mask) == len(self.output_size)  # 1或2

        if random.random() < self.prob:
            image = image[:, ::-1] - np.zeros_like(image)  # TODO 减去全零数组有什么意义
            for i, _output_size in enumerate(self.output_size):
                mask[i] = mask[i][:, ::-1] - np.zeros_like(mask[i])
                joints[i] = joints[i][:, self.flip_index]  # 使用关节翻转索引
                joints[i][:, :, 0] = _output_size - joints[i][:, :, 0] - 1

        return image, mask, joints, area


class RandomAffineTransform(object):
    def __init__(self,
                 input_size,
                 output_size,
                 max_rotation,
                 min_scale,
                 max_scale,
                 scale_type,
                 max_translate):
        self.input_size = input_size
        self.output_size = output_size if isinstance(output_size, list) \
            else [output_size]

        self.max_rotation = max_rotation
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_type = scale_type
        self.max_translate = max_translate

    # 根据中心点位置center, 尺度参数scale, 分辨率res, 旋转参数rot返回仿射变换矩阵
    def _get_affine_matrix(self, center, scale, res, rot=0):
        """创建仿射变换矩阵

        Args:
            center (ndarry: length(x,y)=2): 中心点位置
            scale (float): 以图像长或宽为标准的尺度参数, 除以200
            res (tuple: length(x,y)=2): 图像分辨率
            rot (float): 旋转参数

        Returns:
            t (ndarry: 3, 3): 仿射变换矩阵
            scale (float): 以图像长或宽为标准的尺度参数. 归一化

        """
        # Generate transformation matrix
        h = 200 * scale
        t = np.zeros((3, 3))
        t[0, 0] = float(res[1]) / h
        t[1, 1] = float(res[0]) / h
        t[0, 2] = res[1] * (-float(center[0]) / h + .5)
        t[1, 2] = res[0] * (-float(center[1]) / h + .5)
        t[2, 2] = 1
        scale = t[0,0]*t[1,1]
        if not rot == 0:
            rot = -rot  # To match direction of rotation from cropping
            rot_mat = np.zeros((3, 3))
            rot_rad = rot * np.pi / 180  # 通过乘pi/180转化为弧度
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
            rot_mat[2, 2] = 1
            # Need to rotate around center
            t_mat = np.eye(3)
            t_mat[0, 2] = -res[1]/2
            t_mat[1, 2] = -res[0]/2
            t_inv = t_mat.copy()
            t_inv[:2, 2] *= -1
            t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
        return t, scale

    # 对关节位置进行仿射变换
    def _affine_joints(self, joints, mat):
        joints = np.array(joints)
        shape = joints.shape
        joints = joints.reshape(-1, 2)
        return np.dot(np.concatenate(
            (joints, joints[:, 0:1]*0+1), axis=1), mat.T).reshape(shape)

    # 返回图像的随机仿射变换
    def __call__(self, image, mask, joints, area):
        assert isinstance(mask, list)
        assert isinstance(joints, list)
        assert len(mask) == len(joints)
        assert len(mask) == len(self.output_size)  # 有几个尺度的输出, 就有几个mask和joints

        height, width = image.shape[:2]

        center = np.array((width/2, height/2))  # 中心点位置
        if self.scale_type == 'long':  # 以200像素为标准尺度, 尺度参数以长边为标准
            scale = max(height, width)/200
            print("###################please modify range")
        elif self.scale_type == 'short':  # 尺度参数以短边为标准
            scale = min(height, width)/200
        else:
            raise ValueError('Unkonw scale type: {}'.format(self.scale_type))
        aug_scale = np.random.random() * (self.max_scale - self.min_scale) \
            + self.min_scale  # 设置随机尺度因子参数, 范围[0.75, 1.5]
        scale *= aug_scale  # 计算尺度参数
        aug_rot = (np.random.random() * 2 - 1) * self.max_rotation  # 设置随机旋转角参数, 范围[-30, 30]

        # 对中心点进行随机平移, 范围[-40, 40]
        if self.max_translate > 0:
            dx = np.random.randint(
                -self.max_translate*scale, self.max_translate*scale)
            dy = np.random.randint(
                -self.max_translate*scale, self.max_translate*scale)
            center[0] += dx
            center[1] += dy

        for i, _output_size in enumerate(self.output_size):  # 多尺度输出会有多张图
            mat_output, _ = self._get_affine_matrix(
                center, scale, (_output_size, _output_size), aug_rot
            )  # 返回仿射变换矩阵
            mat_output = mat_output[:2]  # 2*3矩阵
            mask[i] = cv2.warpAffine(
                (mask[i]*255).astype(np.uint8), mat_output,
                (_output_size, _output_size)
            ) / 255
            mask[i] = (mask[i] > 0.5).astype(np.float32)  # 对mask进行仿射变换

            joints[i][:, :, 0:2] = self._affine_joints(
                joints[i][:, :, 0:2], mat_output
            )  # 对关节点进行仿射变换

        mat_input, final_scale = self._get_affine_matrix(
            center, scale, (self.input_size, self.input_size), aug_rot
        )
        mat_input = mat_input[:2]
        area = area*final_scale  # 计算变换后的人体区域面积
        image = cv2.warpAffine(
            image, mat_input, (self.input_size, self.input_size)
        )  # 对图像进行仿射变换

        return image, mask, joints, area
