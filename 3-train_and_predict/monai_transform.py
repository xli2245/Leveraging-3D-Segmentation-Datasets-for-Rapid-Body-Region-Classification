#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import monai
from monai.transforms import (
	EnsureChannelFirstd,
	Compose,
	LoadImaged,
	Orientationd,
	Spacingd,
	ScaleIntensityRanged,
	SpatialPadd,
)
import copy
import numpy as np
from random import sample
from monai.transforms.transform import MapTransform


class CustomRandSpatialCropSamplesd(MapTransform):
	def __init__(self, keys, roi_size, num_samples=1):
		self.keys = keys
		self.roi_size = roi_size
		self.num_samples = num_samples

	def __call__(self, data):
		res = []
		dim, x, y = data[self.keys[0]].shape

		# check the candidates for x dimension
		if x >= self.roi_size[0]:
			candidate_x = [i for i in range(0, x - self.roi_size[0] + 1)]
		else:
			raise ValueError(f'The image size ({x}, {y}) is smaller than roi_size {self.roi_size}')

		# check the candidates for y dimension
		if y >= self.roi_size[1]:
			candidate_y = [i for i in range(0, y - self.roi_size[1] + 1)]
		else:
			raise ValueError(f'The image size ({x}, {y}) is smaller than roi_size {self.roi_size}')

		# combination of x and y
		candidates = []
		while(len(candidates) < self.num_samples):
			cur_x = sample(candidate_x, 1)[0]
			cur_y = sample(candidate_y, 1)[0]
			candidates.append([cur_x, cur_y])

		# do the sampling according to the position of (x, y) in candidates.
		for i in range(self.num_samples):
			x_idx, y_idx = candidates[i][0], candidates[i][1]
			tmp_data = copy.deepcopy(data)
			tmp_img = tmp_data[self.keys[0]]
			tmp_label = tmp_data[self.keys[1]]

			start_x, end_x = x_idx, x_idx+self.roi_size[0]
			start_y, end_y = y_idx, y_idx+self.roi_size[1]

			sampled_img= tmp_img[:, start_x:end_x, start_y:end_y]
			sampled_label = tmp_label[0, start_x:end_x, start_y:end_y, :]
			sampled_label = sampled_label.permute(2, 0, 1)

			tmp_data[self.keys[0]] = sampled_img
			tmp_data[self.keys[1]] = sampled_label

			res.append(tmp_data)
		return res


class CustomSwipeAxis(MapTransform):
	def __init__(self, keys, tag):
		self.keys = keys
		self.tag = tag

	def __call__(self, data):
		tmp_label = data[self.keys[0]]
		if self.tag == 'val':
			sampled_label = tmp_label[0, :, :, :]
			sampled_label = sampled_label.permute(2, 0, 1)
		elif self.tag == 'test':
			sampled_label = tmp_label.permute(0, 3, 1, 2)
		data[self.keys[0]] = sampled_label
		return data


class getDataInfo(MapTransform):
	def __init__(self, keys):
		self.keys = keys

	def __call__(self, data):
		img = data[self.keys[0]]
		label = data[self.keys[1]]
		print('within get data info')
		print(img.size())
		print(label.size())
		return data


class TwoDimProjection(MapTransform):
	def __init__(self, keys, projection_dim, num_classes):
		self.keys = keys
		self.projection_dim = projection_dim
		self.num_classes = num_classes

	def __call__(self, data):
		# project the 3D images into 2D through a certain view by averaging the HU value.
		img = data[self.keys[0]]
		if self.projection_dim == 'axial':
			img = np.mean(img, axis=1)
		elif self.projection_dim == 'coronal':
			img = np.mean(img, axis=2)
		elif self.projection_dim == 'sagittal':
			img = np.mean(img, axis=3)
		else:
			raise ValueError(f'Invalid projection dimension {self.projection_dim}')
		data[self.keys[0]] = torch.from_numpy(img)
		
		if len(self.keys) > 1:
			# project the 3D label eg (1, 128, 132, 64) with integers into multi-hot vector eg (1, 128, 64, 105). 105 is the number of different labels
			label = data[self.keys[1]]
			channel, x, y, z = label.shape
			if self.projection_dim == 'axial':
				new_label = np.zeros((channel, y, z, self.num_classes))
				for j in range(y):
					for k in range(z):
						unique_values = np.unique(label[0, :, j, k])
						new_label[0, j, k, unique_values.astype(int)] = 1
			elif self.projection_dim == 'coronal':
				new_label = np.zeros((channel, x, z, self.num_classes))
				for i in range(x):
					for k in range(z):
						unique_values = np.unique(label[0, i, :, k])
						new_label[0, i, k, unique_values.astype(int)] = 1
			elif self.projection_dim == 'sagittal':
				new_label = np.zeros((channel, x, y, self.num_classes))
				for i in range(x):
					for j in range(y):
						unique_values = np.unique(label[0, i, j, :])
						new_label[0, i, j, unique_values.astype(int)] = 1
			else:
				raise ValueError(f'Invalid projection dimension {self.projection_dim}')
			data[self.keys[1]] = torch.from_numpy(new_label)
		return data


def get_transforms(args):
	""" define transformation for train and dev data
	return: train_transforms, dev_transforms
	"""
	train_transforms = Compose(
		[
			LoadImaged(keys=["image", "label"]),
			EnsureChannelFirstd(keys=["image", "label"]),
			# sample {num_samples} patches with given size {args.patch_size}
			CustomRandSpatialCropSamplesd(
				keys=["image", "label"],
				roi_size=args.patch_size,
				num_samples=16, # 16 for original data
				),
		]
	)

	dev_transforms = Compose(
		[
			LoadImaged(keys=["image", "label"]),
			EnsureChannelFirstd(keys=["image", "label"]),
			CustomSwipeAxis(keys=["label"], tag='val'),
		]
	)
	return train_transforms, dev_transforms


class CustomRotation(MapTransform):
	def __init__(self, keys):
		self.keys = keys

	def __call__(self, data):
		img = data[self.keys[0]]
		img = torch.rot90(img, 1, dims=[1, 2])
		img[0, :, :] = torch.fliplr(img[0, :, :])
		data[self.keys[0]] = img

		if len(self.keys) > 1:
			label = data[self.keys[1]]
			label = torch.rot90(label, 1, dims=[1, 2])
			label[0, :, :, :] = torch.fliplr(label[0, :, :, :])
			data[self.keys[1]] = label
		return data


def get_test_transforms_nii_label(args):
	test_transforms = Compose(
		[
			LoadImaged(keys=["image", "label"]),
			EnsureChannelFirstd(keys=["image", "label"]),
			# make sure the orientation is RAS
			Orientationd(keys=["image", "label"], axcodes="RAS"),
			Spacingd(
				keys=["image", "label"],
				pixdim=(1.5, 1.5, 1.5),
				mode=["nearest", "nearest"],
			),
			# scale/normalize the image intensity, (x - a_min) / (a_max - a_min), clipped if the scaled value outside the range [b_min, b_max]
			ScaleIntensityRanged(keys=["image"], a_min=-1000.0, a_max=2000.0, b_min=0.0, b_max=1.0, clip=True),
			# pad the images and labels with constant value 0 symmetrically if the first or third dimension is less than 64
			SpatialPadd(keys=["image", "label"], spatial_size=(64, 1, 64), method="symmetric", mode="constant", value=0),
			# conduct 2D projection for the images and labels
			TwoDimProjection(keys=["image", "label"], projection_dim='coronal', num_classes=args.num_classes),
			# rotation and flip for input image to make the projected image look exactly as the one shown in itksnap
			CustomRotation(keys=["image", "label"]),
			# swipe axis to ensure label channel first (1, 128, 64, 105) -> (1, 105, 128, 64)
			CustomSwipeAxis(keys=["label"], tag='test'),
		]
	)
	return test_transforms


def get_test_transforms_nii(args):
	test_transforms = Compose(
		[
			LoadImaged(keys=["image"]),
			EnsureChannelFirstd(keys=["image"]),
			# make sure the orientation is RAS
			Orientationd(keys=["image"], axcodes="RAS"),
			Spacingd(
				keys=["image"],
				pixdim=(1.5, 1.5, 1.5),
				mode=["nearest"],
			),
			# scale/normalize the image intensity, (x - a_min) / (a_max - a_min), clipped if the scaled value outside the range [b_min, b_max]
			ScaleIntensityRanged(keys=["image"], a_min=-1000.0, a_max=2000.0, b_min=0.0, b_max=1.0, clip=True),
			# pad the images and labels with constant value 0 symmetrically if the first or third dimension is less than 64
			SpatialPadd(keys=["image"], spatial_size=(64, 1, 64), method="symmetric", mode="constant", value=0),
			# conduct 2D projection for the images and labels
			TwoDimProjection(keys=["image"], projection_dim='coronal', num_classes=args.num_classes),
			# rotation and flip for input image to make the projected image look exactly as the one shown in itksnap
			CustomRotation(keys=["image"]),
		]
	)
	return test_transforms


def get_test_transforms_processed_npy_label(args):
	test_transforms = Compose(
		[
			LoadImaged(keys=["image", "label"]),
			# getDataInfo(keys=["image", "label"]),
			EnsureChannelFirstd(keys=["image", "label"]),
			# getDataInfo(keys=["image", "label"]),
			# swipe axis to ensure label channel first
			CustomSwipeAxis(keys=["label"], tag='test'),
			# getDataInfo(keys=["image", "label"]),
		]
	)
	return test_transforms


def get_test_transforms_processed_npy(args):
	test_transforms = Compose(
		[
			LoadImaged(keys=["image"]),
			EnsureChannelFirstd(keys=["image"]),
		]
	)
	return test_transforms

