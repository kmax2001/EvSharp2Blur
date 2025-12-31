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

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_module import HighResolutionModule
from .conv_block import BasicBlock, Bottleneck, AdaptBlock
from einops import rearrange
import numbers

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck,
    'ADAPTIVE': AdaptBlock
}


class PoseHigherResolutionNet(nn.Module):
    def __init__(self, cfg, modality = None, **kwargs):
        super(PoseHigherResolutionNet, self).__init__()

        modal_dim = 0
        if 'event' in modality:
            modal_dim += 10
            
        if any([(k in modality) for k in ['sharp', 'blur2blur', 'blurred', 'blur']]):
            modal_dim += 3
        self.conv1 = nn.Conv2d(modal_dim, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        # build stage
        self.spec = cfg.MODEL.SPEC
        self.stages_spec = self.spec.STAGES
        self.num_stages = self.spec.STAGES.NUM_STAGES
        num_channels_last = [256]
        for i in range(self.num_stages):
            num_channels = self.stages_spec.NUM_CHANNELS[i]
            transition_layer = \
                self._make_transition_layer(num_channels_last, num_channels)
            setattr(self, 'transition{}'.format(i+1), transition_layer)

            stage, num_channels_last = self._make_stage(
                self.stages_spec, i, num_channels, True
            )
            setattr(self, 'stage{}'.format(i+2), stage)

        # build head net
        inp_channels = int(sum(self.stages_spec.NUM_CHANNELS[-1]))
        config_heatmap = self.spec.HEAD_HEATMAP
        config_offset = self.spec.HEAD_OFFSET
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.num_offset = self.num_joints * 2
        self.num_joints_with_center = self.num_joints+1
        self.offset_prekpt = config_offset['NUM_CHANNELS_PERKPT']
        
        offset_channels = self.num_joints*self.offset_prekpt
        self.transition_heatmap = self._make_transition_for_head(
                    inp_channels, config_heatmap['NUM_CHANNELS'])
        self.transition_offset = self._make_transition_for_head(
                    inp_channels, offset_channels)
        self.head_heatmap = self._make_heatmap_head(config_heatmap)
        self.offset_feature_layers, self.offset_final_layer = \
            self._make_separete_regression_head(config_offset)

        self.pretrained_layers = self.spec.PRETRAINED_LAYERS

    def _make_transition_for_head(self, inplanes, outplanes):
        transition_layer = [
            nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(True)
        ]
        return nn.Sequential(*transition_layer)

    def _make_heatmap_head(self, layer_config):
        heatmap_head_layers = []

        feature_conv = self._make_layer(
            blocks_dict[layer_config['BLOCK']],
            layer_config['NUM_CHANNELS'],
            layer_config['NUM_CHANNELS'],
            layer_config['NUM_BLOCKS'],
            dilation=layer_config['DILATION_RATE']
        )
        heatmap_head_layers.append(feature_conv)

        heatmap_conv = nn.Conv2d(
            in_channels=layer_config['NUM_CHANNELS'],
            out_channels=self.num_joints_with_center,
            kernel_size=self.spec.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if self.spec.FINAL_CONV_KERNEL == 3 else 0
        )
        heatmap_head_layers.append(heatmap_conv)
        
        return nn.ModuleList(heatmap_head_layers)

    def _make_separete_regression_head(self, layer_config):
        offset_feature_layers = []
        offset_final_layer = []

        for _ in range(self.num_joints):
            feature_conv = self._make_layer(
                blocks_dict[layer_config['BLOCK']],
                layer_config['NUM_CHANNELS_PERKPT'],
                layer_config['NUM_CHANNELS_PERKPT'],
                layer_config['NUM_BLOCKS'],
                dilation=layer_config['DILATION_RATE']
            )
            offset_feature_layers.append(feature_conv)

            offset_conv = nn.Conv2d(
                in_channels=layer_config['NUM_CHANNELS_PERKPT'],
                out_channels=2,
                kernel_size=self.spec.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if self.spec.FINAL_CONV_KERNEL == 3 else 0
            )
            offset_final_layer.append(offset_conv)

        return nn.ModuleList(offset_feature_layers), nn.ModuleList(offset_final_layer)

    def _make_layer(
            self, block, inplanes, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, 
                stride, downsample, dilation=dilation))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)


    def _make_stage(self, stages_spec, stage_index, num_inchannels,
                     multi_scale_output=True):
        num_modules = stages_spec.NUM_MODULES[stage_index]
        num_branches = stages_spec.NUM_BRANCHES[stage_index]
        num_blocks = stages_spec.NUM_BLOCKS[stage_index]
        num_channels = stages_spec.NUM_CHANNELS[stage_index]
        block = blocks_dict[stages_spec['BLOCK'][stage_index]]
        fuse_method = stages_spec.FUSE_METHOD[stage_index]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        y_list = [x]
        
        for i in range(self.num_stages):
            x_list = []
            transition = getattr(self, 'transition{}'.format(i+1))
            for j in range(self.stages_spec['NUM_BRANCHES'][i]):
                if transition[j]:
                    x_list.append(transition[j](y_list[-1]))
                else:
                    x_list.append(y_list[j])
            y_list = getattr(self, 'stage{}'.format(i+2))(x_list)
        
        x0_h, x0_w = y_list[0].size(2), y_list[0].size(3)
        x = torch.cat([y_list[0], \
            F.upsample(y_list[1], size=(x0_h, x0_w), mode='bilinear'), \
            F.upsample(y_list[2], size=(x0_h, x0_w), mode='bilinear'), \
            F.upsample(y_list[3], size=(x0_h, x0_w), mode='bilinear')], 1)

        heatmap = self.head_heatmap[1](self.head_heatmap[0](self.transition_heatmap(x)))

        final_offset = []
        offset_feature = self.transition_offset(x)

        for j in range(self.num_joints):
            final_offset.append(
                self.offset_final_layer[j](
                    self.offset_feature_layers[j](
                        offset_feature[:,j*self.offset_prekpt:(j+1)*self.offset_prekpt])))

        offset = torch.cat(final_offset, dim=1)
        return heatmap, offset

    def init_weights(self, pretrained='', verbose=True):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if hasattr(m, 'transform_matrix_conv'):
                nn.init.constant_(m.transform_matrix_conv.weight, 0)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.transform_matrix_conv.bias, 0)
            if hasattr(m, 'translation_conv'):
                nn.init.constant_(m.translation_conv.weight, 0)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.translation_conv.bias, 0)

        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained, 
                            map_location=lambda storage, loc: storage)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    if name in parameters_names or name in buffers_names:
                        if verbose:
                            logger.info(
                                '=> init {} from {}'.format(name, pretrained)
                            )
                        need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)

class HPEFusionNet(nn.Module):
    def __init__(self, num_joints, num_candidates, cfg):
        super(HPEFusionNet, self).__init__()
        
        # Feature extraction
        self.num_candidates = num_candidates
        self.heatmap_encoder = nn.Sequential(
            nn.Conv2d((num_joints+1)*num_candidates+9, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.heatmap_dim = (num_joints+1)
        self.posemap_encoder = nn.Sequential(
            nn.Conv2d(num_joints*2*num_candidates+9, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.posemap_dim = (num_joints*2)

        # Weight estimation for heatmap fusion
        self.heatmap_weight_layer = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1, stride=1)
        )
        self.posemap_weight_layer = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1, stride=1)
        )

        # Output layer
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, heatmaps, posemaps, event_cnt, flow, image):
        h0, w0 = heatmaps.shape[-2:]
        event_cnt= F.interpolate(event_cnt, size=(h0,w0), mode = 'area')
        image = F.interpolate(image, size=(h0,w0), mode = 'bilinear')
        flow= F.interpolate(flow, size=(h0,w0), mode = 'area')
        heatmap_features = self.heatmap_encoder(torch.cat([heatmaps, event_cnt, flow, image], dim = 1))  # Feature extraction
        posemap_features = self.posemap_encoder(torch.cat([posemaps, event_cnt, flow, image], dim =1))
        heatmaps_weights = torch.sigmoid(self.heatmap_weight_layer(heatmap_features)) 
        posemaps_weights = torch.sigmoid(self.posemap_weight_layer(posemap_features))
        
        # Weighted sum
        fused_heatmap = heatmaps[:,0:self.heatmap_dim] * heatmaps_weights[:,0].unsqueeze(1)
        fused_posemap = posemaps[:,0:self.posemap_dim] * posemaps_weights[:,0].unsqueeze(1)
        for i in range(1, self.num_candidates):
            fused_heatmap += heatmaps[:,i*self.heatmap_dim:(i+1)*self.heatmap_dim] * (1-heatmaps_weights[:,0]).unsqueeze(1)  # Pixel-wise multiplication
            fused_posemap += posemaps[:,i*self.posemap_dim:(i+1)*self.posemap_dim] * (1-posemaps_weights[:,0]).unsqueeze(1)
        
        return fused_heatmap, fused_posemap
    


class HPERefineNet(nn.Module):
    def __init__(self, num_joints, num_candidates, cfg):
        super(HPERefineNet, self).__init__()
        
        # Feature extraction
        self.num_candidates = num_candidates
        self.heatmap_encoder = nn.Sequential(
            nn.Conv2d((num_joints+1)*num_candidates+15, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.heatmap_dim = (num_joints+1)
        self.posemap_encoder = nn.Sequential(
            nn.Conv2d(num_joints*2*num_candidates+15, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.posemap_dim = (num_joints*2)
        
        # build stage
        self.spec = cfg.MODEL.SPEC
        self.stages_spec = self.spec.STAGES
        self.num_stages = self.spec.STAGES.NUM_STAGES
         # build head net
        config_heatmap = self.spec.HEAD_HEATMAP
        config_offset = self.spec.HEAD_OFFSET
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.num_offset = self.num_joints * 2
        self.num_joints_with_center = self.num_joints+1
        self.offset_prekpt = config_offset['NUM_CHANNELS_PERKPT']
        
        offset_channels = self.num_joints*self.offset_prekpt
        self.transition_heatmap = self._make_transition_for_head(
                    128, config_heatmap['NUM_CHANNELS'])
        self.transition_offset = self._make_transition_for_head(
                    128, offset_channels)
        self.head_heatmap = self._make_heatmap_head(config_heatmap)
        self.offset_feature_layers, self.offset_final_layer = \
            self._make_separete_regression_head(config_offset)
        
      
        self.heatmap_dim = (num_joints+1)
        self.posemap_dim = (num_joints*2)

        self.refine_heat = EventImage_ChannelAttentionTransformerBlock(128, num_heads=4, 
                                                                       ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')
        self.refine_pose = EventImage_ChannelAttentionTransformerBlock(128, num_heads=4, 
                                                                       ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')

        # Output layer
        
        self._initialize_weights()

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _make_transition_for_head(self, inplanes, outplanes):
        transition_layer = [
            nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(True)
        ]
        return nn.Sequential(*transition_layer)

    def _make_heatmap_head(self, layer_config):
        heatmap_head_layers = []

        feature_conv = self._make_layer(
            blocks_dict[layer_config['BLOCK']],
            layer_config['NUM_CHANNELS'],
            layer_config['NUM_CHANNELS'],
            layer_config['NUM_BLOCKS'],
            dilation=layer_config['DILATION_RATE']
        )
        heatmap_head_layers.append(feature_conv)

        heatmap_conv = nn.Conv2d(
            in_channels=layer_config['NUM_CHANNELS'],
            out_channels=self.num_joints_with_center,
            kernel_size=self.spec.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if self.spec.FINAL_CONV_KERNEL == 3 else 0
        )
        heatmap_head_layers.append(heatmap_conv)
        
        return nn.ModuleList(heatmap_head_layers)

    def _make_separete_regression_head(self, layer_config):
        offset_feature_layers = []
        offset_final_layer = []

        for _ in range(self.num_joints):
            feature_conv = self._make_layer(
                blocks_dict[layer_config['BLOCK']],
                layer_config['NUM_CHANNELS_PERKPT'],
                layer_config['NUM_CHANNELS_PERKPT'],
                layer_config['NUM_BLOCKS'],
                dilation=layer_config['DILATION_RATE']
            )
            offset_feature_layers.append(feature_conv)

            offset_conv = nn.Conv2d(
                in_channels=layer_config['NUM_CHANNELS_PERKPT'],
                out_channels=2,
                kernel_size=self.spec.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if self.spec.FINAL_CONV_KERNEL == 3 else 0
            )
            offset_final_layer.append(offset_conv)

        return nn.ModuleList(offset_feature_layers), nn.ModuleList(offset_final_layer)

    def _make_layer(
            self, block, inplanes, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, 
                stride, downsample, dilation=dilation))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)
    
    
    def forward(self, heatmaps, posemaps, event, flow, image):
        h0, w0 = heatmaps.shape[-2:]
        event= F.interpolate(event, size=(h0,w0), mode = 'bilinear')
        image = F.interpolate(image, size=(h0,w0), mode = 'bilinear')
        flow= F.interpolate(flow, size=(h0,w0), mode = 'area')
        
        heatmap_features = self.heatmap_encoder(torch.cat([heatmaps, event, flow, image], dim = 1))  # Feature extraction
        posemap_features = self.posemap_encoder(torch.cat([posemaps, event, flow, image], dim =1))
        
        heatmap_refine = self.refine_heat(heatmap_features)
        posemap_refine = self.refine_heat(posemap_features)
        
        heatmap = self.head_heatmap[1](
            self.head_heatmap[0](self.transition_heatmap(heatmap_refine)))

        final_offset = []
        offset_feature = self.transition_offset(posemap_refine)

        for j in range(self.num_joints):
            final_offset.append(
                self.offset_final_layer[j](
                    self.offset_feature_layers[j](
                        offset_feature[:,j*self.offset_prekpt:(j+1)*self.offset_prekpt])))

        offset = torch.cat(final_offset, dim=1)
        return heatmap, offset



##########################################################################

## Event-Image Channel Attention (EICA)
class EventImage_ChannelAttentionTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(EventImage_ChannelAttentionTransformerBlock, self).__init__()

        self.norm1_image = LayerNorm(dim, LayerNorm_type)
        self.norm1_event = LayerNorm(dim, LayerNorm_type)
        self.attn = Mutual_Attention(dim, num_heads, bias)
        # mlp
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    # def forward(self, image, event):
    def forward(self, image):
        # image: b, c, h, w
        # event: b, c, h, w
        # return: b, c, h, w
        # assert image.shape == event.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = image.shape
        fused = image + self.attn(self.norm1_image(image), self.norm1_event(image)) # b, c, h, w
        
        # mlp
        fused = to_3d(fused) # b, h*w, c
        fused = fused + self.ffn(self.norm2(fused))
        fused = to_4d(fused, h, w)

        return fused




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Mutual_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Mutual_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        

    def forward(self, x, y):

        assert x.shape == y.shape, 'The shape of feature maps from image and event branch are not equal!'

        b,c,h,w = x.shape

        q = self.q(x) # image
        k = self.k(y) # event
        v = self.v(y) # event
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)



def get_pose_net(cfg, is_train, **kwargs):
    model = PoseHigherResolutionNet(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED, verbose=cfg.VERBOSE)

    return model
