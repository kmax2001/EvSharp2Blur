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

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.NAME = 'regression'
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0
_C.VERBOSE = True
_C.DIST_BACKEND = 'nccl'
_C.MULTIPROCESSING_DISTRIBUTED = False

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'hrnet_dekr'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.FLOWNET_PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.SPEC = CN(new_allowed=True)
_C.MODEL.MODALITY = []


_C.LOSS = CN()
_C.LOSS.WITH_HEATMAPS_LOSS = True
_C.LOSS.HEATMAPS_LOSS_FACTOR = 1.0

_C.LOSS.WITH_OFFSETS_LOSS = True
_C.LOSS.OFFSETS_LOSS_FACTOR = 1.0

_C.LOSS.SEMI_SUP = False

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'coco_kpt'
_C.DATASET.DATASET_TEST = ''
_C.DATASET.NUM_JOINTS = 17
_C.DATASET.NUM_FRAMES = 5
_C.DATASET.MAX_NUM_PEOPLE = 30
_C.DATASET.TRAIN = 'train2017'
_C.DATASET.SOURCE_TEXT = ''
_C.DATASET.TARGET_TEXT = ''
_C.DATASET.TEST = 'val2017'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.DOMAIN_LIST = ['source', None]
_C.DATASET.OTHERS = []
_C.DATASET.DA_SETTING = False
_C.DATASET.DOMAIN_MODALITY = CN()
_C.DATASET.DOMAIN_MODALITY.SOURCE = []
_C.DATASET.DOMAIN_MODALITY.TARGET = []

# training data augmentation
_C.DATASET.MAX_ROTATION = 30
_C.DATASET.MIN_SCALE = 0.75
_C.DATASET.MAX_SCALE = 1.25
_C.DATASET.SCALE_TYPE = 'short'
_C.DATASET.MAX_TRANSLATE = 40
_C.DATASET.INPUT_SIZE = 512
_C.DATASET.OUTPUT_SIZE = 128
_C.DATASET.FLIP = 0.5

# heatmap generator
_C.DATASET.SIGMA = 2.0
_C.DATASET.CENTER_SIGMA = 4.0
_C.DATASET.BG_WEIGHT = 0.1

# offset generator
_C.DATASET.OFFSET_RADIUS = 4

# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.IMAGES_PER_GPU = 32
_C.TRAIN.SHUFFLE = True
_C.TRAIN.TEACHER_FILE = ''
_C.TRAIN.TEACHER_FILE1 = ''
_C.TRAIN.TEACHER_FILE2 = ''
_C.TRAIN.STUDENT_FILE = ''
_C.TRAIN.TEACHER_MODALITY = []
_C.TRAIN.PSEUDO_THRES = 0.1
_C.TRAIN.KEYPOINT_EACH_THRES = 0.0
_C.TRAIN.REFINE_MODALITY = []

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.IMAGES_PER_GPU = 32

_C.TEST.FLIP_TEST = True
_C.TEST.SCALE_FACTOR = [1]

_C.TEST.MODEL_FILE = ''
_C.TEST.MODEL_FILE1 = ''
_C.TEST.MODEL_FILE2 = ''
_C.TEST.POOL_THRESHOLD1 = 300
_C.TEST.POOL_THRESHOLD2 = 200
_C.TEST.NMS_THRE = 0.15
_C.TEST.NMS_NUM_THRE = 10
_C.TEST.KEYPOINT_THRESHOLD = 0.01
_C.TEST.DECREASE = 1.0
_C.TEST.MODALITY = 'blur_processed' #['gt_processed', 'blur_processed']

_C.TEST.MATCH_HMP = False
_C.TEST.ADJUST_THRESHOLD = 0.05
_C.TEST.MAX_ABSORB_DISTANCE = 75
_C.TEST.GUASSIAN_KERNEL = 6

_C.TEST.LOG_PROGRESS = True

_C.RESCORE = CN()
_C.RESCORE.VALID = True
_C.RESCORE.GET_DATA = False
_C.RESCORE.END_EPOCH = 20
_C.RESCORE.LR = 0.001
_C.RESCORE.HIDDEN_LAYER = 256
_C.RESCORE.BATCHSIZE = 1024
_C.RESCORE.MODEL_FILE = 'model/rescore/final_rescore_coco_kpt.pth'
_C.RESCORE.DATA_FILE = 'data/rescore_data/rescore_dataset_train_coco_kpt'


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if not os.path.exists(cfg.DATASET.ROOT):
        cfg.DATASET.ROOT = os.path.join(
            cfg.DATA_DIR, cfg.DATASET.ROOT
        )

    cfg.MODEL.PRETRAINED = os.path.join(
        cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    )

    if cfg.TEST.MODEL_FILE:
        cfg.TEST.MODEL_FILE = os.path.join(
            cfg.DATA_DIR, cfg.TEST.MODEL_FILE
        )

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
