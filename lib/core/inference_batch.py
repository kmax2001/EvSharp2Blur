# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from dataset.transforms import FLIP_CONFIG
from utils.transforms import up_interpolate
import pdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


def get_locations(output_h, output_w, device):
    shifts_x = torch.arange(
        0, output_w, step=1,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, output_h, step=1,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1)

    return locations


def get_reg_poses(offset, num_joints):
    B, _, h, w = offset.shape
    offset = offset.permute(0, 2, 3,1).reshape(B, h*w, num_joints, 2)
    locations = get_locations(h, w, offset.device)
    locations = locations[None, :, None, :].expand(B, -1, num_joints, -1)
    poses = locations - offset

    return poses


def offset_to_pose(offset, flip=True, flip_index=None):
    B, num_offset, h, w = offset.shape
    num_joints = int(num_offset/2)
    reg_poses = get_reg_poses(offset, num_joints) 

    if flip:
        reg_poses = reg_poses[:, flip_index, :]
        reg_poses[:, :, 0] = w - reg_poses[:, :, 0] - 1

    reg_poses = reg_poses.contiguous().view(B, h*w, 2*num_joints).permute(0, 2, 1)
    reg_poses = reg_poses.contiguous().view(B,-1,h,w).contiguous()

    return reg_poses


def get_multi_stage_outputs(cfg, model, teacher, image, event, with_flip=False, modality = None):
    if modality is None:
        modality = cfg.MODEL.MODALITY
    if 'image' in modality:
        inp = image
        if 'event' in modality:
            inp = torch.cat([inp, event], dim =1)
    elif 'event' in modality:
        inp = event
    #inp = event
    pheatmap, poffset =  model(inp)
    theatmap, toffset = teacher(inp)
    heatmap = torch.cat([pheatmap[:,None,:,:,:], theatmap[:,None,:,:,:]], dim = 1)
    offset = torch.cat([poffset[:,None,:,:,:], toffset[:,None,:,:,:]], dim = 1)

    posemap = offset_to_pose(offset, flip=False) #B, 2* K , H, W 

    if with_flip:
        pass

    return heatmap, posemap


def hierarchical_pool(cfg, heatmap):
    pool1 = torch.nn.MaxPool2d(3, 1, 1)
    pool2 = torch.nn.MaxPool2d(5, 1, 2)
    pool3 = torch.nn.MaxPool2d(7, 1, 3)
    map_size = (heatmap.shape[1]+heatmap.shape[2])/2.0
    if map_size > cfg.TEST.POOL_THRESHOLD1:
        maxm = pool3(heatmap[:, :, :, :])
    elif map_size > cfg.TEST.POOL_THRESHOLD2:
        maxm = pool2(heatmap[:, :, :, :])
    else:
        maxm = pool1(heatmap[:, :, :, :])

    return maxm


def get_maximum_from_heatmap(cfg, heatmap):
    B = heatmap.shape[0]
    maxm = hierarchical_pool(cfg, heatmap)
    maxm = torch.eq(maxm, heatmap).float()
    heatmap = heatmap * maxm
    scores = heatmap.view(B, -1)
    scores, pos_ind = scores.topk(cfg.DATASET.MAX_NUM_PEOPLE)
   

    select_ind = (scores > (cfg.TEST.KEYPOINT_THRESHOLD)).nonzero()
    select_ind_mask = scores> cfg.TEST.KEYPOINT_THRESHOLD
 
    return pos_ind, scores, select_ind_mask


def aggregate_results(
        cfg, heatmap_sum, poses, heatmap, posemap, scale, do_print = False
):
    """
    Get initial pose proposals and aggregate the results of all scale.

    Args: 
        heatmap (Tensor): Heatmap at this scale (1, 1+num_joints, w, h)
        posemap (Tensor): Posemap at this scale (1, 2*num_joints, w, h)
        heatmap_sum (Tensor): Sum of the heatmaps (1, 1+num_joints, w, h)
        poses (List): Gather of the pose proposals [(num_people, num_joints, 3)]
    """


    ratio = cfg.DATASET.INPUT_SIZE*1.0/cfg.DATASET.OUTPUT_SIZE
    reverse_scale = ratio/scale
    B, num_joints, w, h = heatmap.shape  
    num_joints -=1
    heatmap_sum += up_interpolate(
        heatmap,
        size=(int(reverse_scale*w), int(reverse_scale*h)),
        mode='bilinear'
    )
    center_heatmap = heatmap[:,-1:, :, :]
    pose_ind, ctr_score, select_ind_mask= get_maximum_from_heatmap(cfg, center_heatmap) 
    
    posemap = posemap.permute(0, 2, 3, 1).view(B, h*w, -1, 2)
    
    if(do_print):
        print(ctr_score[0:2])
        print(select_ind_mask[0:2])
    pose = reverse_scale*posemap[torch.arange(B).unsqueeze(1), pose_ind]
    ctr_score = ctr_score[...,None].repeat(1,1,num_joints)[..., None]
    poses.append(torch.cat([pose, ctr_score], dim=3))
    
    return heatmap_sum, poses, select_ind_mask

def up_interpolate(x,size,mode='bilinear'):
    H=x.size()[2]
    W=x.size()[3]
    scale_h=int(size[0]/H)
    scale_w=int(size[1]/W)
    inter_x= torch.nn.functional.interpolate(x,size=[size[0]-scale_h+1,size[1]-scale_w+1],align_corners=True,mode='bilinear')
    padd= torch.nn.ReplicationPad2d((0,scale_w-1,0,scale_h-1))
    return padd(inter_x)

def get_maximum_from_heatmap_crosscheck(cfg, heatmap, heatmap_crossed):
    B = heatmap.shape[0]
    maxm = hierarchical_pool(cfg, heatmap)
    maxm = torch.eq(maxm, heatmap).float()
    heatmap = heatmap * maxm
    scores = heatmap.view(B, -1)
    scores_crossed = heatmap_crossed.view(B, -1)
    scores, pos_ind = scores.topk(cfg.DATASET.MAX_NUM_PEOPLE)
    scores_crossed = scores_crossed[torch.arange(B).unsqueeze(1), pos_ind]

    select_ind = (scores > (cfg.TEST.KEYPOINT_THRESHOLD)).nonzero()
    select_ind_mask = scores> cfg.TEST.KEYPOINT_THRESHOLD
    select_ind_mask_crossed = scores_crossed > cfg.TEST.KEYPOINT_THRESHOLD
    return pos_ind, scores, select_ind_mask, scores_crossed, select_ind_mask_crossed

def aggregate_results_crosscheck(
        cfg, heatmap_sum, poses, heatmap, posemap, scale, do_print = False
):
    """
    Get initial pose proposals and aggregate the results of all scale.

    Args: 
        heatmap (Tensor): Heatmap at this scale (1, 1+num_joints, w, h)
        posemap (Tensor): Posemap at this scale (1, 2*num_joints, w, h)
        heatmap_sum (Tensor): Sum of the heatmaps (1, 1+num_joints, w, h)
        poses (List): Gather of the pose proposals [(num_people, num_joints, 3)]
    """


    ratio = cfg.DATASET.INPUT_SIZE*1.0/cfg.DATASET.OUTPUT_SIZE
    reverse_scale = ratio/scale
    B, num_joints, w, h = heatmap.shape  #h: 192, w: 128

    num_joints -=1
    heatmap_sum += up_interpolate(
        heatmap,
        size=(int(reverse_scale*w), int(reverse_scale*h)),
        mode='bilinear'
    )
    new_w = int(reverse_scale*w)
    new_h = int(reverse_scale*h)
    heatmap_crossed = heatmap.reshape(B//2,2, num_joints+1, w, h)[:,[-1,0]]
    center_heatmap_crossed = heatmap_crossed.flatten(0,1)[:,-1]

    center_heatmap = heatmap[:,-1:, :, :]
    pose_ind, ctr_score, select_ind_mask, ctr_score_crossed, select_ind_mask_crossed= get_maximum_from_heatmap_crosscheck(cfg, center_heatmap,center_heatmap_crossed) # (batch, 후보 수), (batch, 후보 수), (batch, 후보 수)-> mask 임     
    posemap = posemap.permute(0, 2, 3, 1).view(B, h*w, -1, 2)

    pose = reverse_scale*posemap[torch.arange(B).unsqueeze(1), pose_ind]
    ctr_score = ctr_score[...,None].repeat(1,1,num_joints)[..., None]
    ctr_score_crossed = ctr_score_crossed[...,None].repeat(1,1,num_joints)[..., None]
    poses.append(torch.cat([pose, ctr_score, ctr_score_crossed], dim=3))
    
    return heatmap_sum, poses, select_ind_mask, select_ind_mask_crossed