# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import glob
import matplotlib.pyplot as plt
import random

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


def plot_mAP(result_dict):
    data = [[],[]]
    for itr in result_dict:
        result=result_dict[itr]
        mAP=result['map']
        data[0].append(itr)
        data[1].append(mAP)
    
    plt.figure(figsize = (18, 8))
    color = [random.random(), random.random(), random.random()]
    plt.plot(data[0], data[1], color = color, linewidth = 1.7)
    plt.title('test mAP vs itr')
    plt.savefig(path_to_png)
    #plt.show()


def run_test(model, cfg, ckpt_dir, distributed):

    print('find all pth:')
    for root, dirs, files in os.walk(ckpt_dir):
        pth_file_list = glob.glob(os.path.join(root, '*.pth'))
        break
    # print(pth_file_list)
    ckpt_dict = {}
    for pth_file in pth_file_list:
        itr_str = os.path.basename(pth_file).split('.')[0].split('_')[-1]
        if itr_str[0]=='0':
            itr = int(itr_str)
            ckpt = pth_file
            ckpt_dict[itr]=ckpt
        elif itr_str=='final':
            if cfg.SOLVER.MAX_ITER not in ckpt_dict:
                itr = cfg.SOLVER.MAX_ITER
                ckpt = pth_file
                ckpt_dict[itr]=ckpt
        else:
            print('file name error !')
            exit()
    print(ckpt_dict)
    
    output_dir = cfg.OUTPUT_DIR
    
    result_dict = {}
    for itr in ckpt_dict:
        ckpt=ckpt_dict[itr]
        print(str(itr), ckpt)
        
        checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
        _ = checkpointer.load(ckpt, use_latest=None)
        
        if distributed:
            model = model.module
        torch.cuda.empty_cache()  # TODO check if it helps
        iou_types = ("bbox",)
        if cfg.MODEL.MASK_ON:
            iou_types = iou_types + ("segm",)
        if cfg.MODEL.KEYPOINT_ON:
            iou_types = iou_types + ("keypoints",)
        output_folders = [None] * len(cfg.DATASETS.TEST)
        dataset_names = cfg.DATASETS.TEST
        if cfg.OUTPUT_DIR:
            for idx, dataset_name in enumerate(dataset_names):
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
                mkdir(output_folder)
                output_folders[idx] = output_folder
        data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
        for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
            result=inference(
                            model,
                            data_loader_val,
                            dataset_name=dataset_name,
                            iou_types=iou_types,
                            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                            device=cfg.MODEL.DEVICE,
                            expected_results=cfg.TEST.EXPECTED_RESULTS,
                            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                            output_folder=output_folder,
                            )
            synchronize()
        result_dict[itr]=result
    plot_mAP(result_dict)

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Testing")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt-dir",
        default="",
        metavar="FILE",
        help="path to ckpt file",
        type=str,
    )
    parser.add_argument(
        "--out-dir",
        default="",
        metavar="FILE",
        help="path to out file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if args.out_dir==None:
        print('args.out_dir is None!')

    output_dir = os.path.join(args.out_dir, "inference", cfg.DATASETS.TEST[0])
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(output_dir, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    if args.ckpt_dir:
        run_test(model, cfg, args.ckpt_dir, args.distributed)


if __name__ == "__main__":
    main()
