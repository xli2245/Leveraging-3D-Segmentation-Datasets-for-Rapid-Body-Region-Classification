#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceCELoss, ContrastiveLoss
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.data import decollate_batch
import torch.nn.functional as F
from calculate_statistics import *

torch.backends.cudnn.benchmark = True

def custom_bce_with_logits_loss(outputs, labels):
	# Apply sigmoid to convert outputs to probabilities
	outputs = torch.sigmoid(outputs)
	# Compute binary cross entropy loss for each pixel
	per_pixel_losses = labels * torch.log(outputs + 1e-10) + (1 - labels) * torch.log(1 - outputs + 1e-10)
	# Average over all pixels and batch dimension
	return -per_pixel_losses.mean()


def process(train_loader, val_loader, device, args):
	# define the model
	model = UNet(
		spatial_dims=2,
		in_channels=1,
		out_channels=args.num_classes,
		channels=(64, 128, 256),
		strides=(2, 2),
		num_res_units=2,
		norm=Norm.BATCH,
		).to(device)

	# define the optimizer
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	epochs = args.epochs
	save_path = args.save_path

	# variables for results save
	global_step = 0
	f1_sum_best = 0.0
	micro_f1_best = 0.0
	global_step_f1_sum_best = 0
	global_step_micro_f1_best = 0
	epoch_loss_values = []
	epoch_val_loss_values = []
	metric_values = []

	# training
	def train():
		model.train()
		epoch_loss = 0
		cnt = 0
		epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
		# iterating data
		for step, batch in enumerate(epoch_iterator):
			cnt += 1
			x, y = (batch["image"].cuda(), batch["label"].cuda())
			logit_map = model(x)
			# Assuming 'outputs' are model's predictions and 'labels' are true labels
			# Both 'outputs' and 'labels' have the shape [batch_size, 105, 64, 64]
			loss = custom_bce_with_logits_loss(logit_map, y)
			loss.backward()
			epoch_loss += loss.item()
			optimizer.step()
			optimizer.zero_grad()
			epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.2f)" % (global_step, epochs, loss*100000))
		average_train_loss = epoch_loss / cnt
		return average_train_loss

	# validation
	def val():
		model.eval()
		epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
		stat = []
		val_loss = 0
		with torch.no_grad():
			for step, batch in enumerate(epoch_iterator_val):
				val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
				# using sliding window to conduct evaluation on the whole image instead patches.
				val_outputs = sliding_window_inference(val_inputs, args.patch_size, 4, model)
				# statistics calculation
				micro_f1, macro_f1 = calculate_statistics(val_outputs, val_labels, tag='val')
				# loss calculation
				cur_val_loss = custom_bce_with_logits_loss(val_outputs, val_labels)
				epoch_iterator_val.set_description(
					"Validate (%d / %d Steps) (loss=%2.1f) (micro_f1=%2.4f) (macro_f1=%2.4f)" % (global_step, epochs, cur_val_loss.item(), micro_f1[-1], macro_f1[-1])
				)
				cur_stat = micro_f1 + macro_f1
				stat.append(cur_stat)
				val_loss += cur_val_loss.item()
			stat = np.array(stat)
			mean_stat = np.mean(stat, axis=0)
			mean_val_loss = val_loss / len(epoch_iterator_val)
		return mean_stat, mean_val_loss

	
	while global_step < epochs:
		global_step += 1
		# training
		train_epoch_loss = train()
		epoch_loss_values.append(train_epoch_loss)

		# validation
		stat_val, val_epoch_loss = val()
		metric_values.append(stat_val)
		epoch_val_loss_values.append(val_epoch_loss)
		micro_f1, macro_f1 = stat_val[3], stat_val[-1]
		f1_sum = micro_f1 + macro_f1

		# save loss and dice_val
		np.save(args.save_path + '/loss_for_train_vis.npy', epoch_loss_values)
		np.save(args.save_path + '/val_loss_for_train_vis.npy', epoch_val_loss_values)
		np.save(args.save_path + '/stat_for_train_vis.npy', metric_values)

		with open(args.save_path + '/' + args.tag + '_log.txt', 'a+') as f:
			f.write("Epoch: {}, train_loss: {}, val_loss: {}, micro F1: {}, F1 sum: {}".format(global_step, train_epoch_loss*100000, val_epoch_loss*100000, micro_f1, f1_sum))
			print(f"\t Epoch: {global_step}, train_loss: {train_epoch_loss*100000:.2f}, val_loss: {val_epoch_loss*100000:.2f}, micro F1: {micro_f1:.4f}, F1 sum: {f1_sum:.4f}")
			
			# save best model
			flag = True
			if f1_sum >= f1_sum_best:
				flag = False
				f1_sum_best = f1_sum
				global_step_f1_sum_best = global_step
				torch.save(model.state_dict(), os.path.join(save_path, "best_f1_sum_model.pth"))
				f.write("Best F1 sum model was saved at epoch {}! F1 sum best: {} at step {}; micro F1 best: {} at step {} \n".format(
					global_step, f1_sum_best, global_step_f1_sum_best, micro_f1_best, global_step_micro_f1_best))
				print(f"\t Epoch: {global_step}, Best F1 sum model was saved! F1 sum best: {f1_sum_best:.4f} at step {global_step_f1_sum_best}; micro F1 best: {micro_f1_best:.4f} at step {global_step_micro_f1_best}")
			if micro_f1 >= micro_f1_best:
				flag = False
				micro_f1_best = micro_f1
				global_step_micro_f1_best = global_step
				torch.save(model.state_dict(), os.path.join(save_path, "best_micro_f1_model.pth"))
				f.write("Best micro F1 model was saved at epoch {}! F1 sum best: {} at step {}; micro F1 best: {} at step {} \n".format(
					global_step, f1_sum_best, global_step_f1_sum_best, micro_f1_best, global_step_micro_f1_best))
				print(f"\t Epoch: {global_step}, Best micro F1 model was saved! F1 sum best: {f1_sum_best:.4f} at step {global_step_f1_sum_best}; micro F1 best: {micro_f1_best:.4f} at step {global_step_micro_f1_best}")
			if flag:
				f.write("Model was Not saved at epoch {}! F1 sum best: {} at step {}; micro F1 best: {} at step {} \n".format(
					global_step, f1_sum_best, global_step_f1_sum_best, micro_f1_best, global_step_micro_f1_best))
				print(f"\t Model was Not saved at epoch {global_step}! F1 sum best: {f1_sum_best:.4f} at step {global_step_f1_sum_best}; micro F1 best: {micro_f1_best:.4f} at step {global_step_micro_f1_best}")
			torch.save(model.state_dict(), os.path.join(save_path, "model_latest.pth"))
		
		# adjust_learning_rate, reduced to be half of the previous one every 200 epochs
		adjust_epoch = 200
		if global_step % adjust_epoch == 0:
			lr = args.lr * (0.5 ** (global_step // adjust_epoch))
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr

	print(f"train completed, best_f1_sum: {f1_sum_best:.4f} at iteration: {global_step_f1_sum_best}, best_micro_f1: {micro_f1_best} at iteration: {global_step_micro_f1_best}")

	# picture the trainin progress, training loss and val loss
	plt.figure("train", (12, 6))
	plt.subplot(1, 2, 1)
	plt.title("Iteration Average Loss")
	x = [(i + 1) for i in range(len(epoch_loss_values))]
	y1 = epoch_loss_values
	y2 = epoch_val_loss_values
	plt.xlabel("Iteration")
	plt.plot(x, y1, "x-", label="train loss")
	plt.plot(x, y2, "+-", label="val loss")

	# picture the validation progress, micro f1 and f1 sum
	plt.subplot(1, 2, 2)
	plt.title("Iteration Average Val F1")
	x = [(i + 1) for i in range(len(metric_values))]
	metric_values = np.array(metric_values)
	y1 = metric_values[:, 3]
	y2 = metric_values[:, 7]
	plt.xlabel("Iteration")
	plt.plot(x, y1, "x-", label="val Micro F1")
	plt.plot(x, y2, "+-", label="val F1 sum")
	plt.savefig(os.path.join(save_path, 'training.png'))