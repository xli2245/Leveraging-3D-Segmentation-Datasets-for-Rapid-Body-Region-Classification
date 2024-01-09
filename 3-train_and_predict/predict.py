#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import copy
import torch
import argparse
import numpy as np
import nibabel as nib
from monai_transform import get_test_transforms_nii_label, get_test_transforms_nii, get_test_transforms_processed_npy_label, get_test_transforms_processed_npy

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference

from calculate_statistics import *


torch.backends.cudnn.benchmark = True

root_dir = '' # set as hyperparameter
parser = argparse.ArgumentParser(description='parameters for 2D CT segmenation project')
parser.add_argument('--data_path', metavar='in_DIR', default='../../totalseg_data', 
                    help='path to dataset')
parser.add_argument('--save_path', metavar='out_DIR', default='../3-training/results/test', 
                    help='path to save results')
parser.add_argument('--num_classes', type=int, default=105, metavar='Classes',
                    help='number of classes (default: 105)')
parser.add_argument('--batch_size', type=int, default=1, metavar='BS',
                    help='batch size (default: 1)')
parser.add_argument('--patch_size', type=tuple, default=(64, 64), metavar='PS',
                    help='patch size (default: (64, 64))')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--tag', type=str, default='best', metavar='T',
                    help='tags of model used for prediction')
parser.add_argument('--statistic', type=int, default=0, metavar='Stat',
                    help='whether the statistic based on the prediction and true label is calculated (default: 0)')
parser.add_argument('--input_type', type=str, default='nifti', metavar='input',
                    help='type of the input file, nifti or npy. However nifti is recommended because it\'s transform concludes data preprocessing.')
args = parser.parse_args()

# environment setup
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_test_dataloader(model, test_ds, case_num, args):
    with torch.no_grad():
        img_path = test_ds['image_meta_dict']['filename_or_obj'][0]
        print(f'processing {case_num}: img path is {img_path}')

        # get input
        val_inputs = test_ds["image"].to(device)
        print(f'input shape: {val_inputs.shape}')

        # prediction
        model_pred = sliding_window_inference(
            val_inputs, args.patch_size, 4, model, overlap=0.9, mode="gaussian"
        )
        
        # process and save the prediction results
        predictions = copy.deepcopy(model_pred)
        predictions = torch.sigmoid(predictions)
        predictions = (predictions > 0.5).float()

        predictions = predictions.cpu().numpy()
        predictions = predictions[0, :, :, :]
        predictions = np.transpose(predictions, (1, 2, 0))
        predictions = predictions.astype(np.uint8)

        print(f'output shape: {predictions.shape}')

        filename = img_path.split('/')[-1]
        np.save(img_path.replace(filename, 'predicion_1013_nii.npy'), predictions)

        # calculate the statistic results
        if args.statistic:
            label = test_ds["label"].to(device)
            if len(test_ds["label"].shape) == 5:
                label = label.squeeze(0)
            assert(len(label.shape) == 4)
            model_pred = model_pred.to(device)
            tp, fp, fn, tn, micro, macro = calculate_statistics(model_pred, label, tag='test', dataset='totalseg')
            print(micro)
            print(macro)


if __name__ == '__main__':
    # get the test dataset
    # ps: the json file should match the model input type 
    if args.input_type == 'nifti':
        datasets = os.path.join(args.data_path, "dataset_2d_pred_rotate_nii.json")
    elif args.input_type == 'npy':
        datasets = os.path.join(args.data_path, "dataset_2d_pred_rotate_npy.json")
    else:
        raise ValueError(f'unknown input data type: {args.input_type}')
    test_files = load_decathlon_datalist(datasets, True, "test")

    # get the correct test transforms
    if args.input_type == 'nifti':
        if args.statistic == 1:
            test_transforms = get_test_transforms_nii_label(args)
        else:
            test_transforms = get_test_transforms_nii(args)
    elif args.input_type == 'npy':
        if args.statistic == 1:
            test_transforms = get_test_transforms_processed_npy_label(args)
        else:
            test_transforms = get_test_transforms_processed_npy(args)

    # load the test data into dataloader
    test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_num=1, cache_rate=1.0, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)

    # initialize model
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=args.num_classes,
        channels=(64, 128, 256),
        strides=(2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
        ).to(device)

    # load model for prediction
    if args.tag == 'best':
        model_name = "best_micro_f1_model.pth"
    else:
        model_name = "model_latest.pth"
    model.load_state_dict(torch.load(os.path.join(args.save_path, model_name)))
    model.eval()

    # emuerate the test dataloader
    i = 0
    for d in test_loader:
        i += 1
        predict_test_dataloader(model, d, i, args)