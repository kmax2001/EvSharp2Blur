# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time

from utils.utils import AverageMeter
import torch
import torchvision.transforms
import numpy as np 
import pdb
from utils.transforms import get_multi_scale_size, get_final_preds
from utils.transforms import resize_align_multi_scale
from core.match_batch import match_pose_to_heatmap
from utils.transforms_batch import get_final_preds, get_multi_scale_size
from inference_batch import get_multi_stage_outputs
from inference_batch import aggregate_results_crosscheck, offset_to_pose
from core.nms_batch import pose_nms_crosscheck






def do_train(cfg, model, data_loader, loss_factory, optimizer, epoch,
             output_dir, tb_log_dir, writer_dict, teacher_model1 = None, teacher_model2 = None, teacher_model3= None):
    logger = logging.getLogger("Training")

    batch_time = AverageMeter()
    data_time = AverageMeter()

    heatmap_loss_meter = AverageMeter()
    offset_loss_meter = AverageMeter()
    pseudo_heatmap_loss_meter = AverageMeter()
    pseudo_offset_loss_meter = AverageMeter()


    
    model.eval()
    teacher_model1.eval()
    teacher_model2.eval()
    teacher_model3.eval()

    end = time.time()
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])    
    for i, (data) in enumerate(data_loader):
    
        data_time.update(time.time() - end)
        image = data['image'].cuda()
        event = data['event'].cuda()
        flow = data["flow"].cuda()
        t_original = data["t_image_original"].numpy()
        t_event_original = data["t_event_original"].numpy()
        t_flow_np = data["t_flow_original"].numpy()
        B = t_original.shape[0]
        if True:
            with torch.no_grad():
            # Pseudo label through mutual masking
                base_size, center, scale_resized = get_multi_scale_size(
                    data['t_image_original'].numpy().transpose(0,2,3,1), cfg.DATASET.INPUT_SIZE, 1.0, 1.0
                )
                image_resized = (data["t_image_resized"]).cuda()
                event_resized = (data["t_event_resized"]).cuda()
                t_flow_resized =(data["t_flow_resized"]).cuda()
                poses =[]
                heatmap_sum = 0
                heatmap1, offset1 = model(torch.cat([image_resized, event_resized], dim = 1))
                
                tmp_heatmap, tmp_offset = teacher_model3(event_resized)
                blur_heatmap, blur_offset = teacher_model2(torch.cat([image_resized, event_resized], dim = 1))
                tmp_heatmap, tmp_offset = teacher_model1(torch.cat([tmp_heatmap, blur_heatmap], dim = 1), torch.cat([tmp_offset, blur_offset], dim = 1), event_resized, t_flow_resized, image_resized)
                
                tmp_heatmap = torch.cat([heatmap1[:,None,:,:,:], tmp_heatmap[:,None,:,:,:]], dim = 1).flatten(0,1)
                tmp_offset = torch.cat([offset1[:,None,:,:,:], tmp_offset[:,None,:,:,:]], dim = 1).flatten(0,1)
                posemap = offset_to_pose(tmp_offset, flip= False)
                heatmap_sum, poses, select_ind_mask, select_ind_mask_crossed = aggregate_results_crosscheck(cfg, heatmap_sum, poses, tmp_heatmap, posemap, 1, do_print= (i%100 ==0) and (epoch%5==0))
                poses, scores, scores_crossed, idx_from = pose_nms_crosscheck(cfg, heatmap_sum, poses, select_ind_mask, select_ind_mask_crossed)
                if True:
                    poses = [match_pose_to_heatmap(cfg, poses[j], heatmap_sum[j:j+1])[0] for j in range(len(poses))]
                
                final_poses = get_final_preds(
                    poses, center, scale_resized, base_size
                )
                res = [data_loader.dataset.generate_heatmap_offset(t_original[idx], t_event_original[idx], final_poses[idx], scores[idx].cpu().numpy(), flow = t_flow_np[idx], crossed_score = scores_crossed[idx].cpu().numpy()) for idx in range(B)]
                t_image, t_event, pseudo_heatmap, t_mask, pseudo_offset, t_offset_weight, t_flow = zip(*res)
                t_image, t_event, pseudo_heatmap, t_mask, pseudo_offset, t_offset_weight, t_flow = \
                    list(t_image), list(t_event), list(pseudo_heatmap), list(t_mask), list(pseudo_offset), list(t_offset_weight) , list(t_flow)
                t_flow = torch.stack(t_flow).cuda()
                t_image = torch.stack(t_image).cuda()
                t_event = torch.stack(t_event).cuda()
                pseudo_heatmap = torch.from_numpy(np.stack(pseudo_heatmap))
                t_mask = torch.from_numpy(np.stack(t_mask))
                pseudo_offset = torch.from_numpy(np.stack(pseudo_offset))
                t_offset_weight = torch.from_numpy(np.stack(t_offset_weight))
        with torch.no_grad():
            t_pheatmap, t_poffset = teacher_model3(t_event)
            s_pheatmap, s_poffset = teacher_model3(event)
            t_heatmap2, t_offset2 = teacher_model2(torch.cat([t_image, t_event], dim = 1))
            s_heatmap2, s_offset2 = teacher_model2(torch.cat([image, event], dim = 1))
        t_pheatmap, t_poffset = teacher_model1(torch.cat([t_pheatmap.detach(), t_heatmap2.detach()], dim = 1), torch.cat([t_poffset.detach(), t_offset2.detach()], dim = 1), t_event, t_flow, t_image)
        s_pheatmap, s_poffset = teacher_model1(torch.cat([s_pheatmap.detach(), s_heatmap2.detach()], dim = 1), torch.cat([s_poffset.detach(), s_offset2.detach()], dim = 1), event, flow, image)
            
        s_gt_heatmap = data["heatmap"].cuda(non_blocking=True)
        s_gt_offset = data["offset"].cuda(non_blocking = True)
        s_mask = data["mask"].cuda(non_blocking=True)
        s_offset_w = data["offset_weight"].cuda(non_blocking=True)

        pseudo_heatmap = pseudo_heatmap.cuda(non_blocking = True)
        pseudo_offset = pseudo_offset.cuda(non_blocking = True)
        t_mask = t_mask.cuda(non_blocking = True)
        t_offset_weight = t_offset_weight.cuda(non_blocking = True)
        # train -> sharp, test -> blur
        heatmap_loss, offset_loss = \
            loss_factory(s_pheatmap, s_poffset, s_gt_heatmap, s_mask, s_gt_offset, s_offset_w)
        
        t_heatmap_loss, t_offset_loss = loss_factory(t_pheatmap, t_poffset, pseudo_heatmap, t_mask, pseudo_offset, t_offset_weight)

        loss = 0
        if heatmap_loss is not None:
            heatmap_loss_meter.update(heatmap_loss.item(), image.size(0))
            loss = loss + heatmap_loss
        if offset_loss is not None:
            offset_loss_meter.update(offset_loss.item(), image.size(0))
            loss = loss + offset_loss
        if t_heatmap_loss is not None:
            pseudo_heatmap_loss_meter.update(t_heatmap_loss.item(), image.size(0))
            loss = loss + t_heatmap_loss
        if t_offset_loss is not None:
            pseudo_offset_loss_meter.update(t_offset_loss.item(), image.size(0))
            loss = loss + t_offset_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        

        if i % cfg.PRINT_FREQ == 0 and cfg.RANK == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  '{heatmaps_loss}{offset_loss}{t_heatmaps_loss}{t_offset_loss}\t' \
                  'Times Left: {left_time:.1f}hours'.format( 
                      epoch, i , len(data_loader),
                      batch_time=batch_time,
                      speed=image.size(0)/batch_time.val,
                      data_time=data_time,
                      heatmaps_loss=_get_loss_info(
                          heatmap_loss_meter, 'heatmaps'),
                      offset_loss=_get_loss_info(offset_loss_meter, 'offset'),
                      t_heatmaps_loss= _get_loss_info(
                          pseudo_heatmap_loss_meter, 't_heatmaps'
                      ),
                      t_offset_loss= _get_loss_info(
                          pseudo_offset_loss_meter, 't_offset'
                      ), left_time = (100-epoch)*batch_time.avg*len(data_loader)/3600.0
                  )
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar(
                'train_heatmap_loss',
                heatmap_loss_meter.val,
                global_steps
            )
            writer.add_scalar(
                'train_offset_loss',
                offset_loss_meter.val,
                global_steps
            )
            writer.add_scalar(
                'train_t_heatmap_loss',
                pseudo_heatmap_loss_meter.val,
                global_steps
            )
            writer.add_scalar(
                'train_t_offset_loss',
                pseudo_offset_loss_meter.val,
                global_steps
            )
            writer_dict['train_global_steps'] = global_steps + 1

    return heatmap_loss_meter.avg + offset_loss_meter.avg+pseudo_heatmap_loss_meter.avg+pseudo_offset_loss_meter.avg

def _get_loss_info(meter, loss_name):
    msg = ''
    msg += '{name}: {meter.val:.3e} ({meter.avg:.3e})\t'.format(
        name=loss_name, meter=meter
    )

    return msg
