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
import numpy as np 
import pdb
from core.selection_inference import offset_to_pose, aggregate_results
from core.nms import pose_nms
from utils.transforms import get_multi_scale_size, get_final_preds
from utils.transforms import resize_align_multi_scale
from core.match import match_pose_to_heatmap


def get_pose(cfg,original_images, original_events, heatmaps, offsets, base_size, center, scale_resized, generator, file_path):
    res= []
    for idx in range(heatmaps.shape[0]):
        poses = []
        heatmap_sum = 0
        
        heatmap = heatmaps[idx:idx+1]
        offset = offsets[idx:idx+1]

        posemap = offset_to_pose(offset, flip= False)
        heatmap_sum, poses = aggregate_results(cfg, heatmap_sum, poses, heatmap, posemap, scale= 1)
        heatmap_avg = heatmap_sum/len(cfg.TEST.SCALE_FACTOR)
        poses, scores = pose_nms(cfg, heatmap_avg, poses)
        if len(poses)==0:
            final_poses = np.zeros((1,14,3))
            scores = np.zeros(1)

        else:
            poses = match_pose_to_heatmap(cfg,poses, heatmap_avg)
            final_poses = get_final_preds(poses, center, scale_resized, base_size)    
            final_poses = [pose for _, pose in sorted(zip(scores, final_poses), reverse=True)]
            final_poses = np.stack(final_poses)
            scores = sorted(scores, reverse=True)
            scores = np.array(scores)
        

        root_name = file_path[idx].split('/')[:-1]
        root_name = os.path.join(*root_name)
        os.makedirs(root_name, exist_ok = True)
        file_name = os.path.join(os.path.dirname(file_path[idx].replace("Annotation/train", "new_dir")), "pseudo_pose",file_path[idx].split("/")[-1].replace(".json", ".npz"))
        np.savez(file_name, poses=final_poses, scores = scores)
        
        res.append(generator(original_images[idx], original_events[idx], final_poses, scores))
    
    return res



def do_train(cfg, model, data_loader, loss_factory, optimizer, epoch,
             output_dir, tb_log_dir, writer_dict, teacher_model1 = None, teacher_model2 = None, teacher_model3= None):
    logger = logging.getLogger("Training")

    batch_time = AverageMeter()
    data_time = AverageMeter()

    heatmap_loss_meter = AverageMeter()
    offset_loss_meter = AverageMeter()
    pseudo_heatmap_loss_meter = AverageMeter()
    pseudo_offset_loss_meter = AverageMeter()


    model.train()
    teacher_model1.eval()
    teacher_model2.eval()
    teacher_model3.eval()
    end = time.time()
    
    for i, data in enumerate(data_loader):
        data_time.update(time.time() - end)
        image = data['image'].cuda()
        event = data['event'].cuda()
        t_image= data['t_image'].cuda()
        t_event = data["t_event"].cuda()
        with torch.no_grad():
            file_name = data['t_name']
            base_size, center, scale_resized = get_multi_scale_size(
                data['t_image_original'][0].numpy().transpose(1,2,0), cfg.DATASET.INPUT_SIZE, 1.0, 1.0
            )
            t_original = data["t_image_original"]
            t_event_original = data["t_event_original"]
            t_image = data["t_image_resized"].cuda()
            t_event = data["t_event_resized"].cuda()
            t_flow = data["t_flow_resized"].cuda()
            
            heatmap1, offset1 = teacher_model3(t_event)
            heatmap2, offset2 = teacher_model2(torch.cat([t_image, t_event], dim = 1))
            pseudo_heatmap, pseudo_poffset = teacher_model1(torch.cat([heatmap1, heatmap2], dim = 1), torch.cat([offset1, offset2], dim = 1), t_event, t_flow, t_image)
                            
            res = get_pose(cfg,t_original,t_event_original, pseudo_heatmap, pseudo_poffset, base_size, center, scale_resized, data_loader.dataset.generate_heatmap_offset, file_path= file_name) 
            t_image, t_event, pseudo_heatmap, t_mask, pseudo_offset, t_offset_weight = zip(*res)
            t_image, t_event, pseudo_heatmap, t_mask, pseudo_offset, t_offset_weight = \
                list(t_image), list(t_event), list(pseudo_heatmap), list(t_mask), list(pseudo_offset), list(t_offset_weight) 
            t_image = torch.stack(t_image).cuda()
            t_event = torch.stack(t_event).cuda()
            pseudo_heatmap = torch.from_numpy(np.stack(pseudo_heatmap))
            t_mask = torch.from_numpy(np.stack(t_mask))
            pseudo_offset = torch.from_numpy(np.stack(pseudo_offset))
            t_offset_weight = torch.from_numpy(np.stack(t_offset_weight))
        
        t_pheatmap, t_poffset = model(torch.cat([t_image, t_event], dim =1))
        pheatmap, poffset = model(torch.cat([image, event], dim = 1))
    
        heatmap = data["heatmap"].cuda(non_blocking=True)
        mask = data["mask"].cuda(non_blocking=True)
        offset = data["offset"].cuda(non_blocking=True)
        offset_w = data["offset_weight"].cuda(non_blocking=True)

        pseudo_heatmap = pseudo_heatmap.cuda(non_blocking = True)
        pseudo_offset = pseudo_offset.cuda(non_blocking = True)
        t_mask = t_mask.cuda(non_blocking = True)
        t_offset_weight = t_offset_weight.cuda(non_blocking = True)
        heatmap_loss, offset_loss = \
            loss_factory(pheatmap, poffset, heatmap, mask, offset, offset_w)
        
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
