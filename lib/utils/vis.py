# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import cv2
import numpy as np
import torchvision

#from dataset import VIS_CONFIG
import pdb
import matplotlib.pyplot as plt
import os
coco_part_labels = [
    'nose', 'eye_l', 'eye_r', 'ear_l', 'ear_r',
    'sho_l', 'sho_r', 'elb_l', 'elb_r', 'wri_l', 'wri_r',
    'hip_l', 'hip_r', 'kne_l', 'kne_r', 'ank_l', 'ank_r'
]
coco_part_idx = {
    b: a for a, b in enumerate(coco_part_labels)
}
coco_part_orders = [
    ('nose', 'eye_l'), ('eye_l', 'eye_r'), ('eye_r', 'nose'),
    ('eye_l', 'ear_l'), ('eye_r', 'ear_r'), ('ear_l', 'sho_l'),
    ('ear_r', 'sho_r'), ('sho_l', 'sho_r'), ('sho_l', 'hip_l'),
    ('sho_r', 'hip_r'), ('hip_l', 'hip_r'), ('sho_l', 'elb_l'),
    ('elb_l', 'wri_l'), ('sho_r', 'elb_r'), ('elb_r', 'wri_r'),
    ('hip_l', 'kne_l'), ('kne_l', 'ank_l'), ('hip_r', 'kne_r'),
    ('kne_r', 'ank_r')
]

crowd_pose_part_labels = [
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
    'head', 'neck'
]
crowd_pose_part_idx = {
    b: a for a, b in enumerate(crowd_pose_part_labels)
}
crowd_pose_part_orders = [
    ('head', 'neck'), ('neck', 'left_shoulder'), ('neck', 'right_shoulder'),
    ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'), ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'), ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'), ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle')
]

VIS_CONFIG = {
    'COCO': {
        'part_labels': coco_part_labels,
        'part_idx': coco_part_idx,
        'part_orders': coco_part_orders
    },
    'CROWDPOSE': {
        'part_labels': crowd_pose_part_labels,
        'part_idx': crowd_pose_part_idx,
        'part_orders': crowd_pose_part_orders
    }
}

def add_joints(image, joints, color, dataset='COCO'):
    part_idx = VIS_CONFIG[dataset]['part_idx']
    part_orders = VIS_CONFIG[dataset]['part_orders']

    def link(a, b, color):
        if part_idx[a] < joints.shape[0] and part_idx[b] < joints.shape[0]:
            jointa = joints[part_idx[a]]
            jointb = joints[part_idx[b]]
            if jointa[2] > 0.02 and jointb[2] > 0.02:
                cv2.line(
                    image,
                    (int(jointa[0]), int(jointa[1])),
                    (int(jointb[0]), int(jointb[1])),
                    color,
                    4
                )
            elif jointa[2] > 0 and jointb[2] > 0:
                cv2.line(
                    image,
                    (int(jointa[0]), int(jointa[1])),
                    (int(jointb[0]), int(jointb[1])),
                    [100,100,100],
                    4
                )

    # add joints
    for joint in joints:
        if joint[2] > 0.02:
            cv2.circle(image, (int(joint[0]), int(joint[1])), 1, color, 8)
        elif joint[2] > 0.02:
            cv2.circle(image, (int(joint[0]), int(joint[1])), 1, [100,100,100], 8)


    # add link
    for pair in part_orders:
        link(pair[0], pair[1], color)

    return image


def save_valid_image(image, gt_joints, joints, scores, file_name, color_map= None, dataset='COCO', masking= False, transform = False):
    if transform:
        image = (image*np.array([0.229,0.224,0.225]).reshape(-1,1,1) + np.array([0.485,0.456,0.406]).reshape(-1,1,1))*255
        image = np.clip(image, 0, 254)
        image = cv2.cvtColor(np.transpose(image,(1,2,0)).astype(np.uint8), cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cmap = plt.get_cmap("jet")
    for idx, person in enumerate(joints):
        if color_map is "relative":
            bgr_colors = [
            [0, 0, 0], [255, 255, 255], [0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255],
            [255, 0, 255], [255, 255, 0], [50, 50, 50], [200, 200, 200], [0, 100, 255], [50, 200, 255],
            [0, 128, 0], [144, 238, 144], [139, 0, 0], [173, 216, 230], [42, 42, 165], [225, 255, 181],
            [128, 0, 128], [230, 230, 250], [128, 128, 0], [255, 85, 0], [0, 25, 102], [147, 20, 255],
            [0, 128, 128], [34, 34, 178], [80, 200, 120], [224, 255, 255], [152, 255, 152], [0, 215, 255]
            ]
            color = bgr_colors[idx]
        elif color_map == "white":
            color = [255, 255, 255]
        else:

            if idx>=len(scores):
                idx= -1
            if len(scores)>0:
                score = max(min(scores[idx], 1), 0)
                color = cmap(score)[:3]
                color = [int(c*255) for c in color]
                if masking and (score < 0.3):
                    color = [0,0,0]
            else:
                score = 0
                color = [0,0,0]
        add_joints(image, person, color, dataset=dataset)
    
    for idx, person in enumerate(gt_joints):
        color = [255, 0, 0]
        #add_joints(image, person, color, dataset=dataset)
    if(file_name is not None):  
       cv2.imwrite(file_name, image)
    return image

def save_valid_image_batch(images, gt_joints, joints, scores, file_name, color_map= None, dataset='COCO', masking= False, transform = False):
    N, H, W = images.shape[:3]
    if transform:
        images = (images*np.array([0.229,0.224,0.225]).reshape(-1,1,1) + np.array([0.485,0.456,0.406]).reshape(-1,1,1))*255
        images = np.clip(images, 0, 254)
        images = cv2.cvtColor(np.transpose(images,(1,2,0)).astype(np.uint8), cv2.COLOR_RGB2BGR)
    total_image = np.zeros(images.shape[1:]).repeat(3,0)
    for i in range(N):
        image = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
        cmap = plt.get_cmap("jet")
        for idx, person in enumerate(joints[i]):
            if color_map is "relative":
                bgr_colors = [
                [0, 0, 0], [255, 255, 255], [0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255],
                [255, 0, 255], [255, 255, 0], [50, 50, 50], [200, 200, 200], [0, 100, 255], [50, 200, 255],
                [0, 128, 0], [144, 238, 144], [139, 0, 0], [173, 216, 230], [42, 42, 165], [225, 255, 181],
                [128, 0, 128], [230, 230, 250], [128, 128, 0], [255, 85, 0], [0, 25, 102], [147, 20, 255],
                [0, 128, 128], [34, 34, 178], [80, 200, 120], [224, 255, 255], [152, 255, 152], [0, 215, 255]
                ]
                color = bgr_colors[idx]
            elif color_map == "white":
                color = [255, 255, 255]
            else:

                if idx>=len(scores):
                    idx= -1
                if len(scores)>0:
                    score = max(min(scores[idx], 1), 0)
                    color = cmap(score)[:3]
                    color = [int(c*255) for c in color]
                    if masking and (score < 0.3):
                        color = [0,0,0]
                else:
                    score = 0
                    color = [0,0,0]
            add_joints(image, person, color, dataset=dataset)
    
        for idx, person in enumerate(gt_joints):
            color = [255, 0, 0]
            #add_joints(image, person, color, dataset=dataset)
        total_image[H*i:H*(i+1)] = image
    if(file_name is not None):  
       cv2.imwrite(file_name, total_image)
    return total_image

def make_tagmaps(image, tagmaps):
    num_joints, height, width = tagmaps.shape
    image_resized = cv2.resize(image, (int(width), int(height)))

    image_grid = np.zeros((height, (num_joints+1)*width, 3), dtype=np.uint8)

    for j in range(num_joints):
        tagmap = tagmaps[j, :, :]
        min = float(tagmap.min())
        max = float(tagmap.max())
        tagmap = tagmap.add(-min)\
                       .div(max - min + 1e-5)\
                       .mul(255)\
                       .clamp(0, 255)\
                       .byte()\
                       .cpu()\
                       .numpy()

        colored_tagmap = cv2.applyColorMap(tagmap, cv2.COLORMAP_JET)
        image_fused = colored_tagmap*0.9 + image_resized*0.1

        width_begin = width * (j+1)
        width_end = width * (j+2)
        image_grid[:, width_begin:width_end, :] = image_fused

    image_grid[:, 0:width, :] = image_resized

    return image_grid


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(
                        ndarr,
                        (int(joint[0]), int(joint[1])),
                        2,
                        [255, 0, 0],
                        2
                    )
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_maps(
        batch_image,
        batch_maps,
        batch_mask,
        file_name,
        map_type='heatmap',
        normalize=True
):
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_maps.size(0)
    num_joints = batch_maps.size(1)
    map_height = batch_maps.size(2)
    map_width = batch_maps.size(3)

    grid_image = np.zeros(
        (batch_size*map_height, (num_joints+1)*map_width, 3),
        dtype=np.uint8
    )

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        maps = batch_maps[i]

        if map_type == 'heatmap':
            image_with_hms = make_heatmaps(image, maps)
        elif map_type == 'tagmap':
            image_with_hms = make_tagmaps(image, maps)

        height_begin = map_height * i
        height_end = map_height * (i + 1)

        grid_image[height_begin:height_end, :, :] = image_with_hms
        if batch_mask is not None:
            mask = np.expand_dims(batch_mask[i].byte().cpu().numpy(), -1)
            grid_image[height_begin:height_end, :map_width, :] = \
                grid_image[height_begin:height_end, :map_width, :] * mask

    cv2.imwrite(file_name, grid_image)

def image_debugging(image):
    image = (image*np.array([0.229,0.224,0.225]).reshape(-1,1,1) + np.array([0.485,0.456,0.406]).reshape(-1,1,1))*255

    cv2.imwrite("test4.png", image.transpose(1,2,0))

def heatmap_debugging(image, heatmaps):
    num_joints, height, width = heatmaps.shape
    image_resized = cv2.resize(image.cpu().numpy(), (int(width), int(height)))
    image_resized = (image_resized*np.array([0.229,0.224,0.225]).reshape(-1,1,1) + np.array([0.485,0.456,0.406]).reshape(-1,1,1))*255
    image_grid_avg_keys = np.zeros((height, 3*width, 3))

    for j in range(num_joints-1):
        heatmap = heatmaps[j, :, :]
        image_grid_avg_keys[:,width:2*width,:] += heatmap[..., None]*100
    image_grid_avg_keys[:,2*width:3*width, :] = heatmaps[num_joints-1, :,:][...,None]
    image_grid_avg_keys[:,0*width:width,:] = image_resized
    cv2.imwrite("test1.png", image_grid_avg_keys)
    return image_grid_avg_keys

def save_debugging(image, pheatmap, poffset, gt_heatmap, mask, gt_offset, gt_offweight, name, transform=True):
    h, w = pheatmap.shape[-2:]
    if transform:
        image = (image*np.array([0.229,0.224,0.225]).reshape(-1,1,1) + np.array([0.485,0.456,0.406]).reshape(-1,1,1))*255
        image = np.clip(image, 0, 254)
    image = cv2.cvtColor(np.transpose(image,(1,2,0)).astype(np.uint8), cv2.COLOR_RGB2BGR)
    image_resized = cv2.resize(image, (int(w), int(h)))
    num_joints = pheatmap.shape[0]-1
    image_grid = np.zeros((h*2, 3*w, 3), dtype=np.float32)
    image_grid[h:2*h,0:w] = 1
    pseudo_mask = mask
    mask[mask.nonzero()] =1
    reversed_mask = 1-mask
    
    for j in range(num_joints):
        # add_joints(image_resized, joints[:, j, :])
        heatmap = gt_heatmap[j, :, :]
        pheatmap_tmp = pheatmap[j,:,:]
        image_grid[0:h, w:w*2, :] += heatmap[...,None]*100 #gt_heatmap (1,2)
        image_grid[0:h, 2*w:w*3, :] +=pheatmap_tmp[...,None]*100 #predicted_heatmap (1,3)
        image_grid[h:2*h,0:w] *= mask[j][...,None] #t_mask (2,1)
    mask = image_grid[h:2*h,0:w]*np.array([1, 0, 0])[None,None,:]*255 
    image_grid[h:2*h, w:2*w] = image_grid[h:2*h,0:w]*image_grid[0:h,w:2*w] #masked gt_heatmap(2,2)
    image_grid[h:2*h, 2*w:3*w, :] = image_grid[h:2*h,0:w]*gt_heatmap[-1][...,None]*254 #center (2,3)
    image_grid[h:2*h,0:w] *=255
    image_grid[0:h,0:w] = image_resized# image
    cv2.imwrite(name, image_grid)

def save_debugging_1(image, pheatmap, poffset, gt_heatmap, gt_offset, gt_offweight, name):
    h, w = pheatmap.shape[-2:]
    image = (image*np.array([0.229,0.224,0.225]).reshape(-1,1,1) + np.array([0.485,0.456,0.406]).reshape(-1,1,1))*255
    image = np.clip(image, 0, 254)
    image = cv2.cvtColor(np.transpose(image,(1,2,0)).astype(np.uint8), cv2.COLOR_RGB2BGR)
    image_resized = cv2.resize(image, (int(w), int(h)))
    num_joints = pheatmap.shape[0]-1
    image_grid = np.zeros((h*2, 3*w, 3), dtype=np.float32)
    image_grid[h:2*h,0:w] = 1
    
    for j in range(num_joints):
        # add_joints(image_resized, joints[:, j, :])
        heatmap = gt_heatmap[j, :, :]
        pheatmap_tmp = pheatmap[j,:,:]
        image_grid[0:h, w:w*2, :] += heatmap[...,None]*100 #gt_heatmap (1,2)
        image_grid[0:h, 2*w:w*3, :] +=pheatmap_tmp[...,None]*100 #predicted_heatmap (1,3)
    image_grid[h:2*h, w:2*w] = gt_heatmap[-1][...,None]*254
    image_grid[h:2*h, 2*w:3*w, :] = pheatmap[-1][...,None]*254 #predicted_center_heatmap(2,3)
    image_grid[h:2*h,0:w] *=255
    image_grid[0:h,0:w] = image_resized# image
    image_grid[0:h,[w,2*w],:] = np.ones(3)[None,None,...]*254

    cv2.imwrite(name, image_grid)
    
def save_debugging_2(image, pheatmap1, pheatmap2, name, pheatmap3, pheatmap4):
    h, w = pheatmap1.shape[-2:]
    image = (image*np.array([0.229,0.224,0.225]).reshape(-1,1,1) + np.array([0.485,0.456,0.406]).reshape(-1,1,1))*255
    image = np.clip(image, 0, 254)
    image = cv2.cvtColor(np.transpose(image,(1,2,0)).astype(np.uint8), cv2.COLOR_RGB2BGR)
    image_resized = cv2.resize(image, (int(w), int(h)))
    num_joints = pheatmap1.shape[0]-1
    image_grid = np.zeros((h*2, 5*w, 3), dtype=np.float32)
    image_grid[h:2*h,0:w] = 1
    
    for j in range(num_joints):
        # add_joints(image_resized, joints[:, j, :])
        heatmap1 = pheatmap1[j, :, :]
        heatmap2 = pheatmap2[j,:,:]
        heatmap3 = pheatmap3[j,:,:]
        heatmap4 = pheatmap4[j,:,:]
        image_grid[0:h, w:w*2, :] += heatmap1[...,None]*100 #gt_heatmap (1,2)
        image_grid[0:h, 2*w:w*3, :] +=heatmap2[...,None]*100 #predicted_heatmap (1,3)
        image_grid[0:h, 3*w:4*w,:] += heatmap3[...,None]*100
        image_grid[0:h, 4*w:5*w,:] += heatmap4[...,None]*100
    image_grid[h:2*h, w:2*w] = pheatmap1[-1][...,None]*254
    image_grid[h:2*h, 2*w:3*w, :] = pheatmap2[-1][...,None]*254
    image_grid[h:2*h, 3*w:4*w, :] = pheatmap3[-1][...,None]*254 #event (1,4)
    image_grid[h:2*h, 4*w:5*w, :] = pheatmap4[-1][...,None]*254 #(blur2blur) (1,5)
    image_grid[h:2*h,0:w] *=255
    image_grid[0:h,0:w] = image_resized# image
    image_grid[0:h,[w,2*w,3*w,4*w],:] = np.ones(3)[None,None,...]*254
    cv2.imwrite(name, image_grid)

def save_debug_images(
    config,
    batch_images,
    batch_heatmaps,
    batch_masks,
    batch_outputs,
    prefix
):
    if not config.DEBUG.DEBUG:
        return

    num_joints = config.DATASET.NUM_JOINTS
    batch_pred_heatmaps = batch_outputs[:, :num_joints, :, :]
    batch_pred_tagmaps = batch_outputs[:, num_joints:, :, :]

    if config.DEBUG.SAVE_HEATMAPS_GT and batch_heatmaps is not None:
        file_name = '{}_hm_gt.jpg'.format(prefix)
        save_batch_maps(
            batch_images, batch_heatmaps, batch_masks, file_name, 'heatmap'
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        file_name = '{}_hm_pred.jpg'.format(prefix)
        save_batch_maps(
            batch_images, batch_pred_heatmaps, batch_masks, file_name, 'heatmap'
        )
    if config.DEBUG.SAVE_TAGMAPS_PRED:
        file_name = '{}_tag_pred.jpg'.format(prefix)
        save_batch_maps(
            batch_images, batch_pred_tagmaps, batch_masks, file_name, 'tagmap'
        )

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel