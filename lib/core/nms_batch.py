# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn).
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import pdb

def get_heat_value(pose_coord, heatmap):
    B, _, h, w = heatmap.shape
    heatmap_nocenter = heatmap[:, :-1].flatten(2,3).transpose(1,2)
    y_b = torch.clamp(torch.floor(pose_coord[:,:,:,1]), 0, h-1).long()
    x_l = torch.clamp(torch.floor(pose_coord[:,:,:,0]), 0, w-1).long()
    index = (y_b * w + x_l).unsqueeze(-1)  # (B, human, K, 1)
    heatval = torch.gather(heatmap_nocenter, 1, index.squeeze(-1))  # (B, human, K)

    return heatval


def cal_area_2_torch(v):
    w = torch.max(v[..., 0], dim = -1)[0] - torch.min(v[..., 0], dim = -1)[0]
    h = torch.max(v[... ,1], dim = -1)[0] - torch.min(v[..., 1], dim = -1)[0]
    return w * w + h * h


def nms_core(cfg, pose_coord, heat_score, selection_ind_mask):
    B, num_people, num_joints, _ = pose_coord.shape
    B= B//2
    num_people *=2
    pose_coord = pose_coord.reshape(B,-1, num_joints, 2)
    heat_score = heat_score.reshape(B, -1)
    selection_ind_mask = selection_ind_mask.reshape(B, -1)
    pose_area = cal_area_2_torch(pose_coord)[...,None].repeat(1,1,num_people*num_joints)
    pose_area = pose_area.reshape(B, num_people,num_people,num_joints)
    pose_diff = pose_coord[:, None, :, :, :] - pose_coord[:,:,None,:,:]
    pose_dist = pose_diff.pow(2).sum(-1).sqrt()
    pose_thre = cfg.TEST.NMS_THRE * torch.sqrt(pose_area)
    pose_dist = (pose_dist < pose_thre).sum(-1)
    nms_pose = pose_dist > cfg.TEST.NMS_NUM_THRE
    
    keep_pose_inds = torch.full((B, num_people), False, dtype=torch.bool, device=pose_coord.device)
    indx_list = []
    for b in range(B):
        ignored_pose_inds = set()
        ignored_pose_inds.update(set((~selection_ind_mask[b]).nonzero()[:,0].tolist()))
        nms_list = selection_ind_mask[b].nonzero()[:,0].tolist()
        for i in (nms_list):

            if i in ignored_pose_inds:
                continue
            keep_inds = torch.where(nms_pose[b, i]*selection_ind_mask[b])[0].tolist()
            keep_scores = heat_score[b, keep_inds]
            ind = torch.argmax(keep_scores)
            keep_ind =keep_inds[ind]
            if keep_ind in ignored_pose_inds:
                continue
            keep_pose_inds[b, keep_ind] = True
            
            ignored_pose_inds.update(set(keep_inds))

    return keep_pose_inds



def pose_nms(cfg, heatmap_avg, poses, select_ind_mask):
    """
    NMS for the regressed poses results.

    Args:
        heatmap_avg (Tensor): Avg of the heatmaps at all scales (2, 1+num_joints, w, h)
        poses (List): Gather of the pose proposals [(num_people, num_joints, 3)]
    """
    scale1_index = sorted(cfg.TEST.SCALE_FACTOR, reverse=True).index(1.0)
    pose_norm = poses[scale1_index]
    max_score = pose_norm[:,:,:,2].amax(dim = (1,2)) if pose_norm.shape[0] else 1
    
    for i, pose in enumerate(poses):
        if i != scale1_index:
            max_score_scale = pose[:,:,2].max() if pose.shape[0] else 1
            pose[:,:,2] = pose[:,:,2]/max_score_scale*max_score*cfg.TEST.DECREASE
    pose_score = torch.cat([pose[:,:,:,2:] for pose in poses], dim=1)
    pose_coord = torch.cat([pose[:,:,:,:2] for pose in poses], dim=1)
    
    if pose_coord.shape[0] == 0:
        return [], []

    B, num_people, num_joints, _ = pose_coord.shape
    heatval = get_heat_value(pose_coord, heatmap_avg)  #두input까지는 동일 
    heat_score = (torch.sum(heatval, dim=2)/num_joints)[:,:]
    pose_score = pose_score*heatval.unsqueeze(-1) #pose_score is pose score(based on center score) * heatval(average of heatmap at pose_coord)
    poses = torch.cat([pose_coord.cpu(), pose_score.cpu()], dim=3)
    poses = poses.reshape(B//2,num_people*2, num_joints,-1)
    keep_pose_inds = nms_core(cfg, pose_coord, heat_score, select_ind_mask)
    heat_score = heat_score.reshape(B//2, -1)
    poses_all = []
    heat_scores_all =[]
    for i in range(B//2):
        pose_tmp = poses[i][keep_pose_inds[i]]
        heat_score_tmp = heat_score[i][keep_pose_inds[i]]
        if keep_pose_inds[i].sum() > cfg.DATASET.MAX_NUM_PEOPLE:
            heat_score_tmp, topk_inds = torch.topk(heat_score_tmp, cfg.DATASET.MAX_NUM_PEOPLE)
            pose_tmp = pose_tmp[topk_inds]
        poses_all.append(pose_tmp)
        heat_scores_all.append(heat_score_tmp)
    scores = [batch[:,:,2].mean(dim=1) for batch in poses_all]
    return poses_all, scores

def pose_nms_crosscheck(cfg, heatmap_avg, poses, select_ind_mask, select_ind_mask_crossed):
    """
    NMS for the regressed poses results.

    Args:
        heatmap_avg (Tensor): Avg of the heatmaps at all scales (2, 1+num_joints, w, h)
        poses (List): Gather of the pose proposals [(num_people, num_joints, 3)]
    """
    scale1_index = sorted(cfg.TEST.SCALE_FACTOR, reverse=True).index(1.0)
    pose_norm = poses[scale1_index]
    max_score = pose_norm[:,:,:,2].amax(dim = (1,2)) if pose_norm.shape[0] else 1
    
    for i, pose in enumerate(poses):
        if i != scale1_index:
            max_score_scale = pose[:,:,2].max() if pose.shape[0] else 1
            pose[:,:,2] = pose[:,:,2]/max_score_scale*max_score*cfg.TEST.DECREASE
    
    pose_score = torch.cat([pose[:,:,:,2:3] for pose in poses], dim=1)
    pose_score_crossed = torch.cat([pose[:,:,:, 3:4] for pose in poses], dim =1)
    pose_coord = torch.cat([pose[:,:,:,:2] for pose in poses], dim=1)    
    if pose_coord.shape[0] == 0:
        return [], []

    B, num_people, num_joints, _ = pose_coord.shape
    _, _, w, h = heatmap_avg.shape
    heatval = get_heat_value(pose_coord, heatmap_avg)  
    heatmap_avg_crossed = heatmap_avg.reshape(B//2, 2,1+num_joints,w, h)[:,[-1,0],:].flatten(0,1)
    heatval_crossed = get_heat_value(pose_coord, heatmap_avg_crossed)
    heat_score = (torch.sum(heatval, dim=2)/num_joints)[:,:]
    heat_score_crossed = (torch.sum(heatval_crossed, dim =2)/num_joints)[:,:]

    pose_score = pose_score*heatval.unsqueeze(-1)
    pose_score_crossed = pose_score_crossed*heatval_crossed.unsqueeze(-1)
    poses = torch.cat([pose_coord.cpu(), pose_score.cpu(), pose_score_crossed.cpu()], dim=3)
    poses = poses.reshape(B//2,num_people*2, num_joints,-1)
    keep_pose_inds = nms_core(cfg, pose_coord, heat_score, select_ind_mask)
    heat_score = heat_score.reshape(B//2, -1)
    heat_score_crossed = heat_score_crossed.reshape(B//2,  -1)
    poses_all = []
    heat_scores_all =[]
    heat_scores_crossed_all = []
    idx_from = []
    for i in range(B//2):
        pose_tmp = poses[i][keep_pose_inds[i]]
        idx_from.append(keep_pose_inds[i].nonzero()>= cfg.DATASET.MAX_NUM_PEOPLE)
        heat_score_tmp = heat_score[i][keep_pose_inds[i]]
        heat_score_crossed_tmp = heat_score_crossed[i][keep_pose_inds[i]]
        if keep_pose_inds[i].sum() > cfg.DATASET.MAX_NUM_PEOPLE:
            heat_score_tmp, topk_inds = torch.topk(heat_score_tmp, cfg.DATASET.MAX_NUM_PEOPLE)
            pose_tmp =pose_tmp[topk_inds]
            heat_score_crossed_tmp = heat_score_crossed_tmp[topk_inds]
        poses_all.append(pose_tmp)
        heat_scores_all.append(heat_score_tmp)
        heat_scores_crossed_all.append(heat_score_crossed_tmp)
    scores = [batch[:,:,2].mean(dim=1) for batch in poses_all]
    scores_crossed = [batch[:,:,3].mean(dim=1) for batch in poses_all]
    return poses_all, scores, scores_crossed, idx_from