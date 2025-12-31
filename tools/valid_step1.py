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

import argparse
import os
import sys
import stat
import pprint

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
from tqdm import tqdm
import torch.nn.functional as F
import _init_paths
import models
import numpy as np

from config import cfg
from config import update_config
from core.selection_inference import get_multi_stage_outputs
from core.selection_inference import aggregate_results
from core.nms import pose_nms
from core.match import match_pose_to_heatmap
from dataset import make_test_dataloader
from utils.utils import create_logger
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds, get_affine_transform
from utils.transforms import get_multi_scale_size
from utils.rescore import rescore_valid
from utils.vis import save_valid_image
from dataset.target_generators import HeatmapGenerator, OffsetGenerator

import cv2

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


# markdown format output
def _print_name_value(logger, name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, _ = create_logger(
        cfg, args.cfg, 'valid'
    )

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    event_teacher = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False, modality= ['event']
    )
    blur2blur_teacher = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False, modality= ['blur2blur', 'event']
    )
    model = eval('models.'+cfg.MODEL.NAME+'.HPERefineNet')(
        cfg.DATASET.NUM_JOINTS,2, cfg
    )

    torch.cuda.set_device(1)

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        #model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE, map_location = "cuda")['state_dict'], strict= True)
        event_teacher.load_state_dict(torch.load(cfg.TEST.MODEL_FILE1, map_location = "cuda")['state_dict'], strict=True)
        blur2blur_teacher.load_state_dict(torch.load(cfg.TEST.MODEL_FILE2, map_location = "cuda")['state_dict'], strict=True)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'model_best.pth.tar'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        #model.load_state_dict(torch.load(model_state_file))

    
    event_teacher = event_teacher.cuda()
    blur2blur_teacher = blur2blur_teacher.cuda()
    model = model.cuda()
    event_teacher.eval()
    blur2blur_teacher.eval()
    model.eval()

    data_loader, test_dataset = make_test_dataloader(cfg)
    transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    all_reg_preds = []
    all_reg_scores = []

    
    sigma = cfg.DATASET.SIGMA
    center_sigma = cfg.DATASET.CENTER_SIGMA
    bg_weight = cfg.DATASET.BG_WEIGHT
    pbar = tqdm(total=len(test_dataset)) if cfg.TEST.LOG_PROGRESS else None
    for i, (images, events, image_info, flows, gt_joints) in enumerate(data_loader):
        
        assert 1 == images.size(0), 'Test batch size should be 1'
        image = images[0].cpu().numpy()
        event = events[0].cpu().numpy()
        flow = flows[0].cpu().numpy().transpose(1,2,0)
        event = event.transpose(1,2,0)
        base_size, center, scale = get_multi_scale_size(
            image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0
        )


        with torch.no_grad():
            heatmap_sum = 0
            poses = []
            for scale in sorted(cfg.TEST.SCALE_FACTOR, reverse=True):
                image_resized, center, scale_resized = resize_align_multi_scale(
                    image, cfg.DATASET.INPUT_SIZE, scale, 1.0
                )
                event_resized, _, _ = resize_align_multi_scale(
                    event, cfg.DATASET.INPUT_SIZE, scale, 1.0
                )
                flow_resized, _, _ = resize_align_multi_scale(
                    flow, cfg.DATASET.INPUT_SIZE, scale, 1.0
                )
            
                event_resized = event_resized.transpose(2,0,1)
                event_resized = torch.tensor(event_resized)

                flow_resized = flow_resized.transpose(2,0,1)
                flow_resized = torch.tensor(flow_resized)
                
                image_resized = transforms(image_resized)
                image_resized = image_resized.unsqueeze(0).cuda()
                event_resized = event_resized.unsqueeze(0).cuda()
                flow_resized = flow_resized.unsqueeze(0).cuda()
                heatmap, posemap = get_multi_stage_outputs(
                    cfg, model, image_resized, event_resized, flow_resized, cfg.TEST.FLIP_TEST, modality = ['event'], teacher1 = event_teacher, teacher2 = blur2blur_teacher
                )

                heatmap_sum, poses = aggregate_results(
                    cfg, heatmap_sum, poses, heatmap, posemap, scale
                )
            heatmap_avg = heatmap_sum/len(cfg.TEST.SCALE_FACTOR)
            poses, scores = pose_nms(cfg, heatmap_avg, poses)
            if len(scores) == 0:
                all_reg_preds.append([])
                all_reg_scores.append([])
            else:
                if cfg.TEST.MATCH_HMP:
                    poses = match_pose_to_heatmap(cfg, poses, heatmap_avg)
                final_poses = get_final_preds(
                    poses, center, scale_resized ,base_size 
                )
                if cfg.RESCORE.VALID:
                    scores = rescore_valid(cfg, final_poses, scores)
                all_reg_preds.append(final_poses)
                all_reg_scores.append(scores)
        if cfg.TEST.LOG_PROGRESS:
            pbar.update()

        if True:
            file_name = image_info['file_name'][0]
            prefix = '{}_{}_{}'.format(os.path.join(final_output_dir, 'result_valid'), i, file_name.split('test/')[-1].split('/')[0])

            # logger.info('=> write {}'.format(prefix))
            save_valid_image(image, gt_joints.squeeze(0), final_poses, scores, '{}.jpg'.format(prefix),color_map = None, dataset='CROWDPOSE', masking=True)



    sv_all_preds = [all_reg_preds]
    sv_all_scores = [all_reg_scores]
    sv_all_name = [cfg.NAME]

    if cfg.TEST.LOG_PROGRESS:
        pbar.close()
        
    
    for i in range(len(sv_all_preds)):
        print('Testing '+sv_all_name[i])
        preds = sv_all_preds[i]
        scores = sv_all_scores[i]
        if cfg.RESCORE.GET_DATA:
            test_dataset.evaluate(
                cfg, preds, scores, final_output_dir, sv_all_name[i]
            )
            print('Generating dataset for rescorenet successfully')
        else:
            name_values, _ = test_dataset.evaluate(
                cfg, preds, scores, final_output_dir, sv_all_name[i]
            )

            if isinstance(name_values, list):
                for name_value in name_values:
                    _print_name_value(logger, name_value, cfg.MODEL.NAME)
            else:
                _print_name_value(logger, name_values, cfg.MODEL.NAME)


if __name__ == '__main__':
    main()
