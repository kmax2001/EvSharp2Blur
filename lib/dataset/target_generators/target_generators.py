# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pdb

class HeatmapGenerator():
    def __init__(self, output_res, num_joints):
        self.output_res = output_res
        self.num_joints_with_center = num_joints+1

    def get_heat_val(self, sigma, x, y, x0, y0):

        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        return g

    def __call__(self, joints, sgm, ct_sgm, bg_weight=1.0, mask= None, mask_each = False):
        assert self.num_joints_with_center == joints.shape[1], \
            'the number of joints should be %d' % self.num_joints_with_center
        #mask는 일단 사람마다 적용 
        hms = np.zeros((self.num_joints_with_center, self.output_res[0], self.output_res[1]),
                       dtype=np.float32)
        ignored_hms = 2*np.ones((self.num_joints_with_center, self.output_res[0], self.output_res[1]),
                                    dtype=np.float32)

        hms_list = [hms, ignored_hms]

        for idx_p, p in enumerate(joints):
            for idx, pt in enumerate(p):
                if idx < 17:
                    sigma = sgm
                else:
                    sigma = ct_sgm
                if (pt[2] > 0) and (mask_each == False):
                    x, y = pt[0], pt[1]
                    if (x < 0 or y < 0 or x >= self.output_res[1] or y >= self.output_res[0]):
                        continue

                    ul = int(np.floor(x - 3 * sigma - 1)
                             ), int(np.floor(y - 3 * sigma - 1))
                    br = int(np.ceil(x + 3 * sigma + 2)
                             ), int(np.ceil(y + 3 * sigma + 2))

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res[1])
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res[0])

                    joint_rg = np.zeros((bb-aa, dd-cc))
                    for sy in range(aa, bb):
                        for sx in range(cc, dd):
                            joint_rg[sy-aa, sx -
                                     cc] = self.get_heat_val(sigma, sx, sy, x, y)

                    hms_list[0][idx, aa:bb, cc:dd] = np.maximum(
                        hms_list[0][idx, aa:bb, cc:dd], joint_rg)
                    hms_list[1][idx, aa:bb, cc:dd] = 1.
                    if mask is not None:
                        try:
                            mask[idx_p]
                        except:
                            print(idx)
                            print(idx_p)
                            print(mask)
                        if mask[idx_p] == True:
                            hms_list[1][idx,aa:bb,cc:dd] = 0
                    if mask_each:
                        if joints[idx_p][idx,2] == 0:
                            hms_list[1][idx,aa:bb,cc:dd] = 0



        hms_list[1][hms_list[1] == 2] = bg_weight

        return hms_list


class OffsetGenerator():
    def __init__(self, output_h, output_w, num_joints, radius):
        self.num_joints_without_center = num_joints
        self.num_joints_with_center = num_joints+1
        self.output_w = output_w
        self.output_h = output_h
        self.radius = radius

    def __call__(self, joints, area, mask= None, mask_each = False):
        assert joints.shape[1] == self.num_joints_with_center, \
            'the number of joints should be 18, 17 keypoints + 1 center joint.'

        offset_map = np.zeros((self.num_joints_without_center*2, self.output_h, self.output_w),
                              dtype=np.float32)
        weight_map = np.zeros((self.num_joints_without_center*2, self.output_h, self.output_w),
                              dtype=np.float32)
        area_map = np.zeros((self.output_h, self.output_w),
                            dtype=np.float32)
        for person_id, p in enumerate(joints):
            ct_x = int(p[-1, 0])
            ct_y = int(p[-1, 1])
            ct_v = int(p[-1, 2])
            if (ct_v < 1 or ct_x < 0 or ct_y < 0 or ct_x >= self.output_w or ct_y >= self.output_h):
                continue

            for idx, pt in enumerate(p[:-1]):
                if (pt[2] > 0) or (mask_each) :
                    x, y = pt[0], pt[1]
                    if (x < 0 or y < 0 or x >= self.output_w or y >= self.output_h):
                        continue

                    start_x = max(int(ct_x - self.radius), 0)
                    start_y = max(int(ct_y - self.radius), 0)
                    end_x = min(int(ct_x + self.radius), self.output_w)
                    end_y = min(int(ct_y + self.radius), self.output_h)

                    for pos_x in range(start_x, end_x):
                        for pos_y in range(start_y, end_y):
                            offset_x = pos_x - x
                            offset_y = pos_y - y
                            if offset_map[idx*2, pos_y, pos_x] != 0 \
                                    or offset_map[idx*2+1, pos_y, pos_x] != 0:
                                if area_map[pos_y, pos_x] < area[person_id]:
                                    if mask is not None:
                                        weight_map[idx*2:idx*2+2, pos_y, pos_x] *= (1-mask[person_id])
                                    continue
                            offset_map[idx*2, pos_y, pos_x] = offset_x
                            offset_map[idx*2+1, pos_y, pos_x] = offset_y
                            weight_map[idx*2, pos_y, pos_x] = 1. / (np.sqrt(area[person_id])+0.00001)
                            weight_map[idx*2+1, pos_y, pos_x] = 1. / (np.sqrt(area[person_id])+0.00001)
                            if mask is not None:
                                weight_map[idx*2:idx*2+2, pos_y, pos_x] *= (1-mask[person_id])
                            if mask_each:
                                if joints[person_id][idx, 2] == 0:
                                    weight_map[idx*2:idx*2+2, pos_y, pos_x] *= 0
                            area_map[pos_y, pos_x] = area[person_id]

        return offset_map, weight_map
