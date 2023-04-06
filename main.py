# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for Simple Continual Learning datasets
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------

import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

import timm
from timm.optim import create_optimizer

from datasets import build_continual_dataloader
from engine import *

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

def get_args_parser():
    parser = argparse.ArgumentParser('Simple continual learning training and evaluation configs', add_help=False)
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.03)')
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size per device')
    parser.add_argument('--epochs', default=1, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_tiny_patch16_224', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--pretrained', default=True, help='Load pretrained model or not')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.0, metavar='PCT', help='Drop path rate (default: 0.)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adam"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: (0.9, 0.999), use opt default)')
    parser.add_argument('--clip-grad', type=float, default=1.0, metavar='NORM',  help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay (default: 0.0)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT', help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT', help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')

    # Data parameters
    parser.add_argument('--data-path', default='/local_datasets/', type=str, help='dataset path')
    parser.add_argument('--dataset', default='Split-CIFAR100', type=str, help='dataset name')
    parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle the data order')
    parser.add_argument('--label_shuffle', action='store_true', default=False, help='shuffle the label order when split the single dataset')
    parser.add_argument('--output_dir', default='./output', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # Continual learning parameters
    parser.add_argument('--num_tasks', default=10, type=int, help='number of sequential tasks')
    parser.add_argument('--train_mask', action='store_true', default=True, help='if using the class mask at training')
    parser.add_argument('--no_train_mask', action='store_false', dest='train_mask', help='if domain incremental setting, not using the class mask at training')
    parser.add_argument('--task_inc', action='store_true', default=False, help='if doing task incremental')
    parser.add_argument('--domain_inc', action='store_true', default=False, help='if doing domain incremental')

    # Misc parameters
    parser.add_argument('--print_freq', type=int, default=10, help = 'The frequency of printing')

    return parser


def main(args):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    data_loader, class_mask = build_continual_dataloader(args)

    print(f"Creating model: {args.model}")
    model = timm.create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
    )
    model.to(device)  

    print(args)
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    optimizer = create_optimizer(args, model)

    # must pass the criterion to cuda() to make it work
    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    train_and_evaluate(model, criterion, data_loader, 
                    optimizer, device, class_mask, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Simple continual learning training and evaluation configs', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    sys.exit(0)
