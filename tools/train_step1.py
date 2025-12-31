# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tensorboardX import SummaryWriter

import _init_paths
import models

from config import cfg
from config import update_config
from core.loss import MultiLossFactory
from core.step1_trainer import do_train
from dataset import make_dataloader
from utils.utils import create_logger
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import setup_logger
from utils.utils import get_model_summary
import pdb

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # distributed training
    parser.add_argument('--gpu',
                        help='gpu id for multiprocessing training',
                        type=str)
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    cfg.defrost()
    cfg.RANK = args.rank
    cfg.freeze()

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train'
    )

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or cfg.MULTIPROCESSING_DISTRIBUTED

    ngpus_per_node = torch.cuda.device_count()
    if cfg.MULTIPROCESSING_DISTRIBUTED:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, args, final_output_dir, tb_log_dir)
        )
    else:
        # Simply call main_worker function
        main_worker(
            ','.join([str(i) for i in cfg.GPUS]),
            ngpus_per_node,
            args,
            final_output_dir,
            tb_log_dir
        )


def main_worker(
        gpu, ngpus_per_node, args, final_output_dir, tb_log_dir
):
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if cfg.MULTIPROCESSING_DISTRIBUTED:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        print('Init process group: dist_url: {}, world_size: {}, rank: {}'.
              format(args.dist_url, args.world_size, args.rank))
        dist.init_process_group(
            backend=cfg.DIST_BACKEND,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )

    update_config(cfg, args)

    # setup logger
    logger, _ = setup_logger(final_output_dir, args.rank, 'train')

    selection_model = eval('models.'+cfg.MODEL.NAME+'.HPERefineNet')(
        cfg.DATASET.NUM_JOINTS,2, cfg
    )
    event_subteacher= eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, modality=['event'], is_train= False)
    blurry_event_subteacher= eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, modality=['sharp', 'event'], is_train= False)

    # copy model file
    if not cfg.MULTIPROCESSING_DISTRIBUTED or (
            cfg.MULTIPROCESSING_DISTRIBUTED
            and args.rank % ngpus_per_node == 0
    ):
        this_dir = os.path.dirname(__file__)
        shutil.copy2(
            os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
            final_output_dir
        )

    writer_dict = {
        'writer': SummaryWriter(logdir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            selection_model.cuda(args.gpu)
            event_subteacher.cuda(args.gpu)
            blurry_event_subteacher.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            # args.workers = int(args.workers / ngpus_per_node)
            selection_model = torch.nn.parallel.DistributedDataParallel(
                selection_model, device_ids=[args.gpu]
            )
            event_subteacher = torch.nn.parallel.DistributedDataParallel(
                event_subteacher, device_ids = [args.gpu]
            )
            blurry_event_subteacher = torch.nn.parallel.DistributedDataParallel(
                blurry_event_subteacher, device_ids = [args.gpu]
            )
        else:
            event_subteacher.cuda()
            selection_model.cuda()
            blurry_event_subteacher.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            selection_model = torch.nn.parallel.DistributedDataParallel(selection_model)
            event_subteacher = torch.nn.parallel.DistributedDataParallel(event_subteacher)
            blurry_event_subteacher = torch.nn.parallel.DistributedDataParallel(blurry_event_subteacher)

    elif args.gpu is not None:
        torch.cuda.set_device(int(args.gpu))
        event_subteacher = event_subteacher.cuda(int(args.gpu))
        blurry_event_subteacher = blurry_event_subteacher.cuda(int(args.gpu))
        selection_model = selection_model.cuda(int(args.gpu))
    else:
        event_subteacher = torch.nn.DataParallel(event_subteacher).cuda()
        selection_model = torch.nn.DataParallel(selection_model).cuda()
        blurry_event_subteacher = torch.nn.DataParallel(blurry_event_subteacher).cuda()

    '''
    if not cfg.MULTIPROCESSING_DISTRIBUTED or (
            cfg.MULTIPROCESSING_DISTRIBUTED
            and args.rank % ngpus_per_node == 0
    ):
        dump_input = torch.rand(
            (1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE)
        ).cuda()
        #writer_dict['writer'].add_graph(model, (dump_input, ))
        logger.info(get_model_summary(model, dump_input, verbose=cfg.VERBOSE))
    '''

    # define loss function (criterion) and optimizer
    loss_factory = MultiLossFactory(cfg).cuda()

    # Data loading code
    train_loader = make_dataloader(
        cfg, is_train=True, distributed=args.distributed
    )
    logger.info(train_loader.dataset)

    best_perf = -1
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, selection_model)

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth.tar')
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file, 
                        map_location=lambda storage, loc: storage)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        selection_model.load_state_dict(checkpoint['state_dict'], strict=True)

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))
    
    checkpoint_file = cfg.TRAIN.TEACHER_FILE
    checkpoint_file1 = cfg.TRAIN.TEACHER_FILE1
    logger.info("=> loading teacher net '{}'".format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file, map_location = lambda storage, loc: storage)
    checkpoint1 = torch.load(checkpoint_file1, map_location = lambda storage, loc: storage)
    blurry_event_subteacher.load_state_dict(checkpoint['state_dict'], strict =True)
    event_subteacher.load_state_dict(checkpoint1['state_dict'], strict= True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )
    logger.info("=> loaded")

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        # train one epoch
        do_train(cfg, selection_model, train_loader, loss_factory, optimizer, epoch,
                 final_output_dir, tb_log_dir, writer_dict, ev_teacher= event_subteacher, im_ev_teacher= blurry_event_subteacher)

        lr_scheduler.step()

        perf_indicator = epoch
        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        if not cfg.MULTIPROCESSING_DISTRIBUTED or (
                cfg.MULTIPROCESSING_DISTRIBUTED
                and args.rank == 0
        ):
            if epoch % 5 ==0:
                logger.info('=> saving checkpoint to {}'.format(final_output_dir))
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': cfg.MODEL.NAME,
                    'state_dict': selection_model.state_dict(),
                    # 'best_state_dict': model.module.state_dict(),
                    'best_state_dict': selection_model.state_dict(),
                    'perf': perf_indicator,
                    'optimizer': optimizer.state_dict(),
                }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state{}.pth.tar'.format(gpu)
    )

    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(selection_model.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
