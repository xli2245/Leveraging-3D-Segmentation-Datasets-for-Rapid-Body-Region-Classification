#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import argparse
from process import process
from monai_transform import get_transforms

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

torch.backends.cudnn.benchmark = True

root_dir = '' # set as hyperparameter
parser = argparse.ArgumentParser(description='parameters for 2D classification using data projected from 3D')
parser.add_argument('--data_path', type=str, default='../../totalseg_data', metavar='data DIR',
                    help='path to dataset')
parser.add_argument('--save_path', type=str, default='./results/test', metavar='output DIR', 
                    help='path to save results')
parser.add_argument('--num_classes', type=int, default=105, metavar='Classes', 
                    help='num of classes for classification (default: 105)')
parser.add_argument('--epochs', type=int, default=400, metavar='EPOCH', 
                    help='epochs (default: 400)')
parser.add_argument('--lr', default=0.001, type=float, metavar='LR', 
                    help='initial learning rate')
parser.add_argument('--weight_decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-2)')
parser.add_argument('--batch_size', '--bs', type=int, default=1,
                    help='batch size (default: 1)')
parser.add_argument('--patch_size', '--ps', type=tuple, default=(64, 64), 
                    help='patch_size (default: (64, 64))')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--tag', type=str, default='train', metavar='T',
                    help='tags for each training')
args = parser.parse_args()

# environment setup
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
datasets = os.path.join(args.data_path, "data_split_processed.json")
train_files = load_decathlon_datalist(datasets, True, "train")
val_files = load_decathlon_datalist(datasets, True, "dev")

# get transforms
train_transforms, val_transforms = get_transforms(args)

train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_num=96, cache_rate=1.0, num_workers=16)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=240, cache_rate=1.0, num_workers=16)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

# start training
process(train_loader, val_loader, device, args)

