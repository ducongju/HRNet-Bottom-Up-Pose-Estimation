# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class HeatmapGenerator():
    def __init__(self, output_res, num_joints):
        self.output_res = output_res
        self.num_joints = num_joints

    def get_heat_val(self, sigma, x, y, x0, y0):

        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        return g

    # 关节点坐标joints, [3,17], 关节点方差sgm,
    # 中心点方差ct_sgm, 背景权重bg_weight
    def __call__(self, joints, sgm, ct_sgm, bg_weight=1.0):
        assert self.num_joints == joints.shape[1], \
            'the number of joints should be %d' % self.num_joints

        hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                       dtype=np.float32)  # 生成关节热图hms, ndarry: (17, 128, 128)
        ignored_hms = 2*np.ones((1, self.output_res, self.output_res),
                                dtype=np.float32)  # 生成背景热图ignored_hms, ndarry: (1, 128, 128)

        hms_list = [hms, ignored_hms]  # list: 2

        for p in joints:
            for idx, pt in enumerate(p):
                if idx < 17:
                    sigma = sgm
                    # TODO PIFPAF
                else:
                    sigma = ct_sgm  # 为关节点和中心点
                if pt[2] > 0:
                    x, y = pt[0], pt[1]
                    if x < 0 or y < 0 or \
                            x >= self.output_res or y >= self.output_res:
                        continue  # 关节点位置在图外, 跳过该注释

                    # TODO 把R修改为自适应, 这里的R固定为15*15的正方形边界
                    ul = int(np.floor(x - 3 * sigma - 1)
                             ), int(np.floor(y - 3 * sigma - 1))  # TODO 为什么要减一
                    br = int(np.ceil(x + 3 * sigma + 2)
                             ), int(np.ceil(y + 3 * sigma + 2))  # 通过3sigma原则得到的关节点左上角和右下角

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)  # 关节点的边界框不能超出图像外

                    joint_rg = np.zeros((bb-aa, dd-cc))  # 设置关节点的边界框
                    for sy in range(aa, bb):
                        for sx in range(cc, dd):
                            joint_rg[sy - aa, sx - cc] = self.get_heat_val(sigma, sx, sy, x, y)  # 边界框内添加高斯核

                    hms_list[0][idx, aa:bb, cc:dd] = np.maximum(
                        hms_list[0][idx, aa:bb, cc:dd], joint_rg)  # 关节热图的边界框内的值不能会负数, 且重叠位置使用最大值
                    hms_list[1][0, aa:bb, cc:dd] = 1.  # 对应背景热图所有关节点的边界框内的值为1

        hms_list[1][hms_list[1] == 2] = bg_weight  # 设置所有人的所有关节以外的背景热图值为bg_weight=0.1, tradeoff loss

        return hms_list


class OffsetGenerator():
    def __init__(self, output_h, output_w, num_joints, radius):
        self.num_joints_without_center = num_joints - 1
        self.output_w = output_w
        self.output_h = output_h
        self.num_joints = num_joints
        self.radius = radius

    def __call__(self, joints, area):
        assert joints.shape[1] == self.num_joints, \
            'the number of joints should be 18, 17 keypoints + 1 center joint.'

        offset_map = np.zeros((self.num_joints_without_center*2, self.output_h, self.output_w),
                              dtype=np.float32)  # 生成偏移图offset_map, ndarry: (17*2, 128, 128)
        weight_map = np.zeros((self.num_joints_without_center*2, self.output_h, self.output_w),
                              dtype=np.float32)  # 生成区域权值图, ndarry: (17*2, 128, 128), 给偏移向量添加权重, 人越大权重越小
        area_map = np.zeros((self.output_h, self.output_w),
                            dtype=np.float32)  # 生成区域面积图, 位于中心点附近的区域具有人体边界框面积的值

        for person_id, p in enumerate(joints):
            ct_x = int(p[-1, 0])
            ct_y = int(p[-1, 1])
            ct_v = int(p[-1, 2])
            if ct_v < 1 or ct_x < 0 or ct_y < 0 \
                    or ct_x >= self.output_w or ct_y >= self.output_h:
                continue

            for idx, pt in enumerate(p[:-1]):
                if pt[2] > 0:
                    x, y = pt[0], pt[1]
                    if x < 0 or y < 0 or \
                            x >= self.output_w or y >= self.output_h:
                        continue

                    start_x = max(int(ct_x - self.radius), 0)
                    start_y = max(int(ct_y - self.radius), 0)
                    end_x = min(int(ct_x + self.radius), self.output_w)
                    end_y = min(int(ct_y + self.radius), self.output_h)

                    # 只在偏移半径范围内才给值
                    for pos_x in range(start_x, end_x):
                        for pos_y in range(start_y, end_y):
                            offset_x = pos_x - x
                            offset_y = pos_y - y
                            if offset_map[idx*2, pos_y, pos_x] != 0 \
                                    or offset_map[idx*2+1, pos_y, pos_x] != 0:
                                if area_map[pos_y, pos_x] < area[person_id]:  # TODO 除了考虑人体尺度，还可考虑距离
                                    continue
                            offset_map[idx*2, pos_y, pos_x] = offset_x
                            offset_map[idx*2+1, pos_y, pos_x] = offset_y
                            weight_map[idx*2, pos_y, pos_x] = 1. / np.sqrt(area[person_id])
                            weight_map[idx*2+1, pos_y, pos_x] = 1. / np.sqrt(area[person_id])
                            area_map[pos_y, pos_x] = area[person_id]

        return offset_map, weight_map
