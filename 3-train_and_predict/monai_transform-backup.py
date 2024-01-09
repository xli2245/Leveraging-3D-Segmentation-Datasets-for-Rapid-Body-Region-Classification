#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import monai
from monai.transforms import (
	EnsureChannelFirstd,
	Orientationd,
	ScaleIntensityRanged,
	RandGaussianNoised,
	RandZoomd,
	RandFlipd,
	RandRotated,
	SpatialPadd,
	Compose,
	LoadImaged,
	SaveImaged,
)
import copy
import scipy
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
			sampled_label = sampled_label.transpose(2, 0, 1)

			tmp_data[self.keys[0]] = sampled_img
			tmp_data[self.keys[1]] = sampled_label

			res.append(tmp_data)
		return res


class CustomSwipeAxis(MapTransform):
	def __init__(self, keys):
		self.keys = keys

	def __call__(self, data):
		tmp_label = data[self.keys[0]]
		sampled_label = tmp_label[0, :, :, :]
		sampled_label = sampled_label.transpose(2, 0, 1)
		data[self.keys[0]] = sampled_label
		return data


class CustomSwipeAxisPred(MapTransform):
	def __init__(self, keys):
		self.keys = keys

	def __call__(self, data):
		tmp_label = data[self.keys[0]]
		sampled_label = tmp_label.transpose(0, 3, 1, 2)
		data[self.keys[0]] = sampled_label
		return data

class CustomSwipeAxisPostProcess(MapTransform):
	def __init__(self, keys):
		self.keys = keys

	def __call__(self, data):
		tmp_label = data[self.keys[0]]
		sampled_label = tmp_label.transpose(0, 1, 3, 2)
		data[self.keys[0]] = sampled_label
		return data


class getDataInfo(MapTransform):
	def __init__(self, keys):
		self.keys = keys

	def __call__(self, data):
		# img = data[self.keys[0]]
		# label = data[self.keys[1]]
		print('get info 1')
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
		
		# project the 3D label eg (1, 128, 128, 64) with integers into multi-hot vector eg (1, 128, 64, 105). 105 is the number of different labels
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
		
		data[self.keys[0]] = img
		data[self.keys[1]] = new_label
		return data


class RandPoissonNoise(MapTransform):
	def __init__(self, keys, scale_factor=0.1):
		self.keys = keys
		self.scale_factor = scale_factor

	def __call__(self, data):
		ori_img= data[self.keys[0]]
		P_lambda = self.scale_factor * np.std(ori_img)
		Poisson_noise = np.random.poisson(lam=P_lambda, size=ori_img.shape)
		noisy_img = ori_img + Poisson_noise
		data[self.keys[0]] = noisy_img
		return data


class CustomRandFlip(MapTransform):
	def __init__(self, keys, prob=0.1, spatial_axis=0):
		self.keys = keys
		self.prob = prob
		self.spatial_axis = spatial_axis
	
	def __call__(self, data):
		ori_img = data[self.keys[0]]
		ori_label = data[self.keys[1]]
		noisy_img = ori_img
		noisy_label = ori_label
		if np.random.rand() < self.prob:
			noisy_img = np.flip(ori_img, axis=self.spatial_axis).copy()
			noisy_label = np.flip(ori_label, axis=self.spatial_axis).copy()
		data[self.keys[0]] = noisy_img
		data[self.keys[1]] = noisy_label
		return data

# random rotate the images and labels with given range_x, range_y, range_z
class CustomRandRotated(MapTransform):
	def __init__(self, keys, range_x, range_y, range_z, prob=0.1, mode=["constant", "constant"], value=[-1000.0, 0.0], keep_size=True):
		self.keys = keys
		self.range_x = (-range_x, range_x) if isinstance(range_x, (int, float)) else range_x
		self.range_y = (-range_y, range_y) if isinstance(range_y, (int, float)) else range_y
		self.range_z = (-range_z, range_z) if isinstance(range_z, (int, float)) else range_z
		self.prob = prob
		self.mode = mode
		self.value = value
		self.keep_size = keep_size
	
	def __call__(self, data):
		# Perform the rotation only with the given probability
		if np.random.rand() < self.prob:
			# Get the image/label to be rotated
			ori_img = data[self.keys[0]]
			ori_label = data[self.keys[1]]

			img = np.squeeze(ori_img, axis=0)
			label = np.squeeze(ori_label, axis=0)
			
			# Generate random angles for x, y, and z axes within the specified ranges
			angle_x = np.random.uniform(self.range_x[0], self.range_x[1])
			angle_y = np.random.uniform(self.range_y[0], self.range_y[1])
			angle_z = np.random.uniform(self.range_z[0], self.range_z[1])
			
			# Perform the rotation (here using scipy's rotate function as an example)
			# For 3D image, the axes to rotate are specified as pairs (plane normal vectors)
			img_rot_x = scipy.ndimage.rotate(img, angle_x, axes=(1, 2), reshape=not self.keep_size, order=0, mode=self.mode[0], cval=self.value[0])
			img_rot_y = scipy.ndimage.rotate(img_rot_x, angle_y, axes=(0, 2), reshape=not self.keep_size, order=0, mode=self.mode[0], cval=self.value[0])
			img_rot_z = scipy.ndimage.rotate(img_rot_y, angle_z, axes=(0, 1), reshape=not self.keep_size, order=0, mode=self.mode[0], cval=self.value[0])

			label_rot_x = scipy.ndimage.rotate(label, angle_x, axes=(1, 2), reshape=not self.keep_size, order=0, mode=self.mode[1], cval=self.value[1])
			label_rot_y = scipy.ndimage.rotate(label_rot_x, angle_y, axes=(0, 2), reshape=not self.keep_size, order=0, mode=self.mode[1], cval=self.value[1])
			label_rot_z = scipy.ndimage.rotate(label_rot_y, angle_z, axes=(0, 1), reshape=not self.keep_size, order=0, mode=self.mode[1], cval=self.value[1])
			
			# Store the rotated image back to the data dictionary
			data[self.keys[0]] = np.expand_dims(img_rot_z, axis=0)
			data[self.keys[1]] = np.expand_dims(label_rot_z, axis=0)
		return data

# random zoom in or zoom out images and labels with given min_zoom and max_zoom
class CustomRandZoomd(MapTransform):
	def __init__(self, keys, prob=0.1, min_zoom=0.8, max_zoom=1.2, mode=["bilinear", "nearest"], padding_mode=["zeros", "zeros"], keep_size=True, value=[-1000, 0]):
		self.keys = keys
		self.prob = prob
		self.min_zoom = min_zoom
		self.max_zoom = max_zoom
		self.mode = mode
		self.padding_mode = padding_mode
		self.keep_size = keep_size
		self.value = value
	
	def __call__(self, data):
		# Perform the zoom only with the given probability
		if np.random.rand() < self.prob:
			# Get the image/label to be rotated
			img = data[self.keys[0]]
			label = data[self.keys[1]]
			
			# Generate random zoom factor
			zoom_factor = np.random.uniform(self.min_zoom, self.max_zoom)
			
			# Perform the zoom (here using scipy's zoom function as an example)
			img_zoom = scipy.ndimage.zoom(img, zoom_factor, mode=self.mode[0], cval=self.value[0], order=0, prefilter=False)
			label_zoom = scipy.ndimage.zoom(label, zoom_factor, mode=self.mode[1], cval=self.value[1], order=0, prefilter=False)
			
			# Pad back to the original size
			if self.keep_size and zoom_factor < 1.0:
				# Calculate padding
				pad_img = []
				pad_label = []
				for i in range(len(img.shape)):
					# Calculate the padding for each dimension
					pad_left = (img.shape[i] - img_zoom.shape[i]) // 2
					pad_right = img.shape[i] - img_zoom.shape[i] - pad_left
					# Store the padding for each dimension
					pad_img.append((pad_left, pad_right))
					pad_label.append((pad_left, pad_right))
				# Pad the image and label
				img_zoom = np.pad(img_zoom, pad_img, mode=self.padding_mode[0], constant_values=self.value[0])
				label_zoom = np.pad(label_zoom, pad_label, mode=self.padding_mode[1], constant_values=self.value[1])
			elif self.keep_size and zoom_factor > 1.0:
				# Calculate the center crop size
				crop_size = img.shape

				# Calculate the starting indices for cropping each dimension
				start_indices = [(img_zoom.shape[i] - crop_size[i]) // 2 for i in range(len(crop_size))]
				
				# Calculate the ending indices for cropping each dimension
				end_indices = [start_indices[i] + crop_size[i] for i in range(len(crop_size))]

				# Perform the crop operation
				if len(crop_size) == 2:
					img_zoom_cropped = img_zoom[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1]]
					label_zoom_cropped = label_zoom[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1]]
				elif len(crop_size) == 3:
					img_zoom_cropped = img_zoom[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_indices[2]:end_indices[2]]
					label_zoom_cropped = label_zoom[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_indices[2]:end_indices[2]]
				elif len(crop_size) == 4:  # For 3D images with channel dimension
					img_zoom_cropped = img_zoom[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_indices[2]:end_indices[2], start_indices[3]:end_indices[3]]
					label_zoom_cropped = label_zoom[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_indices[2]:end_indices[2], start_indices[3]:end_indices[3]]
				else:
					raise ValueError("Unsupported number of dimensions.")
				img_zoom = img_zoom_cropped
				label_zoom = label_zoom_cropped
			
			# Store the zoomed image back to the data dictionary
			data[self.keys[0]] = img_zoom
			data[self.keys[1]] = label_zoom
		return data


def get_test_transforms(args):
	test_transforms = Compose(
		[
			LoadImaged(keys=["image", "label"]),
			EnsureChannelFirstd(keys=["image", "label"]),
			# make sure the orientation is RAS
			Orientationd(keys=["image", "label"], axcodes="RAS"),
			# scale/normalize the image intensity, (x - a_min) / (a_max - a_min), clipped if the scaled value outside the range [b_min, b_max]
			ScaleIntensityRanged(keys=["image"], a_min=-1000.0, a_max=2000.0, b_min=0.0, b_max=1.0, clip=True),
			# pad the images and labels with constant value 0 symmetrically 
			SpatialPadd(keys=["image", "label"], spatial_size=(64, 1, 64), method="symmetric", mode="constant", value=0),
			# conduct 2D projection for the images and labels
			TwoDimProjection(keys=["image", "label"], projection_dim='coronal', num_classes=args.num_classes),
			# swipe axis to ensure label channel first
			CustomSwipeAxisPred(keys=["label"]),
		]
	)
	return test_transforms


def get_transforms(args):
	""" define transformation for train and dev data
	return: train_transforms, dev_transforms
	"""
	train_transforms = Compose(
		[
			LoadImaged(keys=["image", "label"]),
			EnsureChannelFirstd(keys=["image", "label"]),
			# make sure the orientation is RAS
			# Orientationd(keys=["image", "label"], axcodes="RAS"),
			
			# augmentation: RandPoissonNoise for CT and PET, RandGaussianNoise for MRI 
			RandPoissonNoise(keys=['image'], scale_factor=0.05),

			# augmentation: random flip the images and labels (does this work for different orientation?)
			CustomRandFlip(keys=["image", "label"], prob=0.5, spatial_axis=1),
			CustomRandFlip(keys=["image", "label"], prob=0.5, spatial_axis=2),
			CustomRandFlip(keys=["image", "label"], prob=0.5, spatial_axis=3),

			# augmentation: random rotate the images and labels
			CustomRandRotated(keys=["image", "label"], range_x=0, range_y=np.pi/12, range_z=0, prob=1, mode=["constant", "constant"], value=[-1000.0, 0.0], keep_size=True),
			
			# augmentation: random zoom in or zoom out
			CustomRandZoomd(keys=["image", "label"], prob=1, min_zoom=0.8, max_zoom=1.2, mode=["constant", "constant"], padding_mode=["constant", "constant"], keep_size=False, value=[-1000, 0]),

			# SaveImaged(keys=["image"], output_dir='../../totalseg_data/s0000', output_postfix='aug', output_ext='.nii.gz', resample=False, dtype=np.float32),

			# scale/normalize the image intensity, (x - a_min) / (a_max - a_min), clipped if the scaled value outside the range [b_min, b_max]
			ScaleIntensityRanged(keys=["image"], a_min=-1000.0, a_max=2000.0, b_min=0.0, b_max=1.0, clip=True),

			# pad the images and labels with constant value 0 symmetrically 
			SpatialPadd(keys=["image", "label"], spatial_size=(64, 1, 64), method="symmetric", mode="constant", value=0),

			# conduct 2D projection for the images and labels, the dimmension allowed includes 'axial', 'coronal', and 'saggital'.
			TwoDimProjection(keys=["image", "label"], projection_dim='coronal', num_classes=args.num_classes),

			# sample {num_samples} patches with given size {args.patch_size}
			CustomRandSpatialCropSamplesd(
				keys=["image", "label"],
				roi_size=args.patch_size,
				num_samples=16, # 16 for original data
				),
			getDataInfo(keys=["image", "label"]),
			
		]
	)

	dev_transforms = Compose(
		[
			LoadImaged(keys=["image", "label"]),
			EnsureChannelFirstd(keys=["image", "label"]),
			# make sure the orientation is RAS
			# Orientationd(keys=["image", "label"], axcodes="RAS"),
			# scale/normalize the image intensity, (x - a_min) / (a_max - a_min), clipped if the scaled value outside the range [b_min, b_max]
			ScaleIntensityRanged(keys=["image"], a_min=-1000.0, a_max=2000.0, b_min=0.0, b_max=1.0, clip=True),
			# pad the images and labels with constant value 0 symmetrically 
			SpatialPadd(keys=["image", "label"], spatial_size=(64, 1, 64), method="symmetric", mode="constant", value=0),
			# conduct 2D projection for the images and labels
			TwoDimProjection(keys=["image", "label"], projection_dim='coronal', num_classes=args.num_classes),
			# swipe axis to ensure label channel first
			CustomSwipeAxis(keys=["label"]),
		]
	)
	return train_transforms, dev_transforms


def get_post_transforms(keys, output_dir, output_postfix='pred'):
	post_transforms = Compose(
		[
			CustomRandFlip(keys=keys, prob=1, spatial_axis=3),
			CustomRandFlip(keys=keys, prob=1, spatial_axis=2),
			CustomSwipeAxisPostProcess(keys=keys),
			SaveImaged(
				keys=keys,
				output_dir=output_dir,
				output_postfix=output_postfix,
				output_dtype=np.uint8,
				resample=False,
			),
		]
	)
	return post_transforms
