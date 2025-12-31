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

import logging

import numpy as np
import torch

from .CrowdPoseDataset import CrowdPoseDataset
import pdb


import os
import cv2
import json
from torchvision.transforms import functional as F
import torch
import random
import pdb
from dataset import *
from utils.transforms import resize_align_multi_scale
from crowdposetools.coco import COCO

logger = logging.getLogger(__name__)

MODALITY_MAP = {"sharp": 'gt_processed', "blur2blur": "blur2blur_processed", 
                             "blur": "blur_processed", "blurred": "blurred_processed", "event": "event_voxel"}

def binary_search_array(array, x, left=None, right=None, side="left"):
    """
    Binary search through a sorted array.
    """

    left = 0 if left is None else left
    right = len(array) - 1 if right is None else right
    mid = left + (right - left) // 2

    if left > right:
        return left if side == "left" else right

    if array[mid] == x:
        return mid

    if x < array[mid]:
        return binary_search_array(array, x, left=left, right=mid - 1)

    return binary_search_array(array, x, left=mid + 1, right=right)

class CrowdPoseKeypoints(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset, heatmap_generator, offset_generator=None, transforms=None):
        # super().__init__(cfg, dataset)
        super(CrowdPoseKeypoints).__init__()
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.num_joints_with_center = self.num_joints+1

        self.sigma = cfg.DATASET.SIGMA
        self.center_sigma = cfg.DATASET.CENTER_SIGMA
        self.bg_weight = cfg.DATASET.BG_WEIGHT

        self.heatmap_generator = heatmap_generator
        self.offset_generator = offset_generator
        self.transforms = transforms
        
        self.json_dir = cfg.DATASET.ROOT
        self.domain_list = ['source' if len(cfg.DATASET.DOMAIN_MODALITY.SOURCE) else 'None', 
                            'target' if len(cfg.DATASET.DOMAIN_MODALITY.TARGET) else 'None']
        self.modality = {
            'source': 
                {'image': next((k for k in ['sharp', 'blur', 'blurred', 'blur2blur'] if k in cfg.DATASET.DOMAIN_MODALITY.SOURCE), None),
                'event': next((k for k in ['event'] if k in cfg.DATASET.DOMAIN_MODALITY.SOURCE), None)
                },
            'target':
                {'image': next((k for k in ['sharp', 'blur', 'blurred', 'blur2blur'] if k in cfg.DATASET.DOMAIN_MODALITY.TARGET), None),
                'event': next((k for k in ['event'] if k in cfg.DATASET.DOMAIN_MODALITY.TARGET), None)
                }
            }
        self.others = cfg.DATASET.OTHERS #['event_cnt', 'flow']
        self.label_paths  = []
        self.target_paths = []
        self.data_length = 0
        self.source_range = 0
        self.da_setting = cfg.DATASET.DA_SETTING
        self.psudo_thres= cfg.TRAIN.PSEUDO_THRES
        self.keypoint_thres = cfg.TRAIN.KEYPOINT_EACH_THRES
        self.key_mask = self.keypoint_thres>0
        self.resize = True
        self.input_size = cfg.DATASET.INPUT_SIZE
        if ('source' in self.domain_list):
            source_list, length = self.make_dataset_list(cfg.DATASET.SOURCE_TEXT)
            self.label_paths += source_list
            self.data_length += length
            self.source_range += length
        if ('target' in self.domain_list):
            if 'txt' in cfg.DATASET.TARGET_TEXT:
                target_list, length = self.make_dataset_list(cfg.DATASET.TARGET_TEXT)
            else:
                target_list, length = self.make_dataset_list_coco(cfg.DATASET.TARGET_TEXT)
            if self.da_setting:
                self.target_paths += target_list
            else:
                self.label_paths += target_list
                self.data_length += length

        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        
    def __len__(self):
        return self.data_length
        
    def make_dataset_list(self, domain_path):
        parent_folder_list_all = []
        with open(domain_path, 'r') as f:
            lines = f.readlines()
        for l in lines:
            parent_folder_list_all.append(l.strip('\n')) 
        parent_folder_list_all.sort()
        label_paths = []
        image_ids = 0
        for dir_name in parent_folder_list_all:
            dataset_dir = os.path.join(self.json_dir, dir_name)
            if not os.path.isdir(dataset_dir):
                continue
            folder_list_all = os.listdir(dataset_dir)
            folder_list_all.sort()

            for file_name in folder_list_all:
                file_path = os.path.join(dataset_dir, file_name)
                if os.path.isdir(file_path):
                    continue
                label_paths.append(file_path)
                image_ids +=1
        data_length = len(label_paths)
        return label_paths, data_length
    
    def make_dataset_list_coco(self, json_file):
        self.coco = COCO(json_file)
        self.ids = list(self.coco.imgs.keys())
        return self.ids, len(self.ids)
    
    def __getitem__(self, index):
        
        gt_path= self.label_paths[index]
        if index <self.source_range:
            modality = self.modality["source"]
        else:
            #in case we train with target domain data without DA method.
            modality = self.modality["target"]
        
        inp_data = dict()
        inp_data = self.prepare_data(inp_data, gt_path, modality)
        
        if self.da_setting:
            t_gt_path = self.target_paths[index % len(self.target_paths)]
            t_modality = self.modality["target"]
            inp_data = self.prepare_data(inp_data, t_gt_path, t_modality,prefix="t_", add_full=True)
        
        return inp_data
    
    
    def prepare_data(self, inp_data, gt_path, modality, prefix="", add_full=False):
        event_path = gt_path.replace('Annotation/train', 'new_dir').replace('json', 'npz').split('/')[:-1] + \
            [MODALITY_MAP['event']] + gt_path.replace('Annotation/train', 'new_dir').replace('json', 'npz').split('/')[-1:]
        event_path = os.path.join(*event_path)
        f_flow_path = event_path.replace("event_voxel/", "flow/f_")
        b_flow_path = event_path.replace("event_voxel/", "flow/b_")
        used_data = []
        channel_index = [0]

        if modality['image'] is not None:
            img_path = gt_path.replace('Annotation/train', 'new_dir').replace('json', 'png').split('/')[:-1] + \
            [MODALITY_MAP[modality['image']]] + gt_path.replace('Annotation/train', 'new_dir').replace('json', 'png').split('/')[-1:]
            img_path = os.path.join(*img_path)
            used_data.append('image')
            image = cv2.imread(
            img_path,
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            channel_index.append(channel_index[-1]+3)

        if modality['event'] is not None:
            used_data.append('event')
            event = np.load(event_path)['data'].transpose(1,2,0)
            channel_index.append(channel_index[-1]+10)

        if 'flow' in self.others:
            used_data.append('flow')
            f_flow = np.load(f_flow_path)
            b_flow = np.load(b_flow_path)
            flow = np.concatenate([f_flow['flow'].transpose(1,2,0), b_flow['flow'].transpose(1,2,0)], axis = 2)
            f_flow.close()
            b_flow.close()
            channel_index.append(channel_index[-1]+2)
            
        with open(gt_path) as json_data:
            loaded_file = json.load(json_data)
        
        
        width = 1376
        height = 928
        m = np.zeros((height, width))
        
        mask = m < 0.5
    
        datas = eval(used_data[0])
        for data in used_data[1:]:
            datas = np.concatenate([datas, eval(data)], axis = 2)

        object_metas = loaded_file['annotations']
        joints, area = self.get_joints(object_metas)
        
        if add_full:            
            if "image" in used_data:
                #apply resize_align_multi_scale separately from other modality due to different dtype.
                image_resized, center, scale_resized = resize_align_multi_scale(
                    eval("image"), self.input_size, 1, 1.0
                )
                inp_data[prefix+"image_resized"] = image_resized.transpose(2,0,1)
                target_datas = datas[:,:,channel_index[1]:]
                inp_data[prefix+"image_resized"] = F.normalize(torch.from_numpy(inp_data[prefix+"image_resized"])/255.0, mean = self.mean, std=self.std)
            else:
                target_datas = datas
            target_datas_resized, center, scale_resized = resize_align_multi_scale(
                target_datas, self.input_size, 1, 1.0
            )
            for i, data in enumerate(used_data):
                inp_data[prefix+data+"_original"] = eval(data).transpose(2,0,1)
                
                if data =="image":
                    continue                
                if "image" in used_data:
                    inp_data[prefix+data+"_resized"] = target_datas_resized.transpose(2,0,1)[channel_index[i]-3:channel_index[i+1]-3]            
                else:
                    inp_data[prefix+data+"_resized"] = target_datas_resized.transpose(2,0,1)[channel_index[i]:channel_index[i+1]]            
                
        if self.transforms:
            datas, mask_list, joints_list, area = self.transforms(
                datas, [mask], [joints], area
            )

        for i, data in enumerate(used_data):
            inp_data[prefix+data] = datas[channel_index[i]:channel_index[i+1],:,:]
        
        if 'image' in used_data:
            inp_data[prefix+'image'] = F.normalize((inp_data[prefix+'image'])/255.0, mean=self.mean, std=self.std)

        heatmap, ignored = self.heatmap_generator(
            joints_list[0], self.sigma, self.center_sigma, self.bg_weight)
        mask = mask_list[0]*ignored 
        offset, offset_weight = self.offset_generator(
            joints_list[0], area)
        
        inp_data[prefix+"name"] = gt_path
        inp_data[prefix+"heatmap"] = heatmap
        inp_data[prefix+"mask"] = mask
        inp_data[prefix+"offset"] = offset
        inp_data[prefix+"offset_weight"] = offset_weight
        return inp_data
        


    def cal_area_2_torch(self, v):
        w = torch.max(v[:, :, 0], -1)[0] - torch.min(v[:, :, 0], -1)[0]
        h = torch.max(v[:, :, 1], -1)[0] - torch.min(v[:, :, 1], -1)[0]
        return w * w + h * h

    def merge_half_voxel(self, front, back):
        c, h, w = front.shape
        merged = np.zeros((2*c-1,h,w))
        merged[:c] += front
        merged[-c:] += back
        return merged

    def generate_heatmap_offset(self, image, event, poses, scores, key_mask = False, flow = None,crossed_score = None):

        num_people = poses.shape[0]
        joints = np.zeros((num_people, self.num_joints_with_center, 3))

        width = 1376
        height = 928
        m = np.zeros((height, width))
        mask_l = m < 0.5
        area = np.zeros((num_people, 1))
        mask = np.zeros(scores.shape)
        mask = scores < self.psudo_thres
        
        if crossed_score is not None:
            mask = np.logical_or(mask, crossed_score<self.psudo_thres)

        for i in range(num_people):
            area[i,0] = self.cal_area_2_torch(torch.tensor(poses[i:i+1,:,:]))
            poses_sum = np.sum(poses[i, :, :2], axis=0)
            num_vis_joints = len(np.nonzero(poses[i, :, 2])[0])
            if num_vis_joints <=0 :
                joints[i, -1, :] = 0
            else:
                joints[i, -1, :-1] = poses_sum/ num_vis_joints
            joints[i,:-1, :-1] = poses[i,:,:-1]
            joints[i,:-1, 2] = 1.0

        joints[:,-1,2] = 1
        image = np.concatenate([image, event], axis = 0)
        if flow is not None:
            image = np.concatenate([image,flow], axis= 0)
        
        image = image.transpose(1,2,0)
        if self.transforms:
            image, mask_list, joints_list, area = self.transforms(
                image, [mask_l], [joints], area
            )
        event = image[3:13]
        if flow is not None:
            flow = image[13:15]
        image = image[:3]
        image = F.normalize(image/255.0, mean=self.mean, std = self.std)
        heatmap, ignored = self.heatmap_generator(joints_list[0], self.sigma, self.center_sigma, self.bg_weight, mask= mask)
        mask_l = mask_list[0]*ignored
        offset, offset_weight = self.offset_generator(joints_list[0], area, mask = mask)
        if flow is not None:
            return image, event, heatmap, mask_l, offset, offset_weight, flow        
        return image, event, heatmap, mask_l, offset, offset_weight

    def get_joints(self, anno):
        num_people = len(anno)
        area = np.zeros((num_people, 1))
        joints = np.zeros((num_people, self.num_joints_with_center, 3))

        for i in range(num_people):
            key_points = anno[i]['keypoints'][0]
            if len(key_points) == 0:
                continue
            
            for k_ipt in range(len(key_points)):
                x_coord = float(anno[i]['keypoints'][0][k_ipt][0])
                y_coord = float(anno[i]['keypoints'][0][k_ipt][1])
                
                joints[i, k_ipt, 0] = x_coord
                joints[i, k_ipt, 1] = y_coord
                
                if x_coord != 0 or y_coord != 0:
                    joints[i, k_ipt, 2] = 1
                else:
                    joints[i, k_ipt, 2] = 0
            area[i, 0] = self.cal_area_2_torch(
                torch.tensor(joints[i:i+1,:,:]))

            joints_sum = np.sum(joints[i, :-1, :2], axis=0)
            num_vis_joints = len(np.nonzero(joints[i, :-1, 2])[0])
            if num_vis_joints <= 0:
                joints[i, -1, :2] = 0
            else:
                joints[i, -1, :2] = joints_sum / num_vis_joints
            joints[i, -1, 2] = 1

        return joints, area

    def get_joints_from_pseudo(self, anno, mask=True):
        num_people = len(anno)
        area = np.zeros((num_people, 1))
        joints = np.zeros((num_people, self.num_joints_with_center, 3))
        masking = []
        scores = []
        for i in range(num_people):
            key_points = anno[i]['keypoints']
            score = anno[i]['score']
            scores.append(score)
            masking.append(score>self.psudo_thres)
            if len(key_points) == 0:
                continue
            
            for k_ipt in range((self.num_joints)):
                x_coord = float(anno[i]['keypoints'][3*k_ipt+ 0])
                y_coord = float(anno[i]['keypoints'][3*k_ipt+1])
                
                joints[i, k_ipt, 0] = x_coord
                joints[i, k_ipt, 1] = y_coord
                joints[i, k_ipt, 2] = (anno[i]['keypoints'][3*k_ipt+2])>0
                
            area[i, 0] = self.cal_area_2_torch(
                torch.tensor(joints[i:i+1,:,:]))
            sum_mask  =joints[i,:,2]>0
            joints_sum = np.sum(joints[i, sum_mask, :2], axis=0)
            num_vis_joints = anno[i]['num_keypoints']
            if num_vis_joints <= 0:
                joints[i, -1, :2] = 0
            else:
                joints[i, -1, :2] = joints_sum / num_vis_joints
            joints[i, -1, 2] = 1

        return joints, area, masking, scores

    def get_pseudo_joints(self, file_path, crosscheck= None):
        file = np.load(file_path)
        poses = file["poses"]
        scores = file["scores"]
        mask = np.zeros(scores.shape)
        mask = scores < self.psudo_thres
        if crosscheck is not None and crosscheck ==True :
            crossed_score = file["crossed_scores"]
            mask = np.logical_or(mask, crossed_score<self.psudo_thres)
        num_people = poses.shape[0]

        joints = np.zeros((num_people, self.num_joints_with_center, 3))
        masking_keypoint = np.zeros((num_people, self.num_joints_with_center))
        width = 1376
        height = 928
        m = np.zeros((height, width))
        mask_l = m < 0.5
        area = np.zeros((num_people, 1))

        for i in range(num_people):
            area[i,0] = self.cal_area_2_torch(torch.tensor(poses[i:i+1,:,:]))
            num_vis_joints = len(np.nonzero(poses[i, :, 2])[0])
            mask_idx = poses[i][:,2] > 0.0
            poses_sum = np.sum(poses[i][mask_idx, :2], axis=0)
            num_vis_joints = mask_idx.sum()
            no_pose = (num_people==1) and (poses[i].any() ==False)
            if self.key_mask:
                mask_idx = (poses[i][:,2] > self.keypoint_thres)

                poses_sum = np.sum(poses[i][mask_idx,:2],axis=0)
                num_vis_joints = mask_idx.sum()


            if num_vis_joints <=0 : 
                joints[i, -1, :-1] = 0
                if no_pose:
                    joints[i,-1,-1] = 0
                else:
                    joints[i,-1, 2] = 1
            else:
                joints[i, -1, :-1] = poses_sum/ num_vis_joints
                joints[i, :, 2] = 1 
            joints[i,:-1, :2] = poses[i,:,:2]
            if self.key_mask:
                joints[i,:-1, 2][mask_idx==0] = 0.0 
                

        return joints, area, mask, scores
        

    def get_mask(self, anno, img_info):
        m = np.zeros((img_info['height'], img_info['width']))

        return m < 0.5
    
    
