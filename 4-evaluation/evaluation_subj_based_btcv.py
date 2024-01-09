#!/usr/bin/python
# -*- coding: UTF-8 -*-


import json
import os
import torch
import numpy as np

num = 0.5

def process_subj(pred_path, label_path):
	# load the npy file
	label = np.load(label_path)
	pred = np.load(pred_path)

	# align the pred and label
	pred = np.squeeze(pred, axis=0)
	pred = np.transpose(pred, (1, 2, 0))

	label = np.squeeze(label, axis=0)
	label = np.transpose(label, (1, 2, 0))

	# sigmoid
	pred = torch.from_numpy(pred)
	pred = torch.sigmoid(pred)
	pred = pred.numpy()
	pred[pred >= num] = 1
	pred[pred < num] = 0

	pred = pred.astype(int)
	label = label.astype(int)

	# squeeze into channels
	label_bool = np.any(label == 1, axis=(0, 1))
	pred_bool = np.any(pred == 1, axis=(0, 1))

	label_int = label_bool.astype(int)
	pred_int = pred_bool.astype(int)

	# label_list = list(label_int)
	# pred_list = list(pred_int)

	return pred_int, label_int


def count_statistics(pred, label):
	# Compute the metrics
	TP = np.logical_and(pred == 1, label == 1).astype(int)
	TN = np.logical_and(pred == 0, label == 0).astype(int)
	FP = np.logical_and(pred == 1, label == 0).astype(int)
	FN = np.logical_and(pred == 0, label == 1).astype(int)

	# Stack the results into a 2D array
	metrics = np.stack((TP, TN, FP, FN), axis=-1)
	return metrics


def compute_metrics(metrics):
    # specific for BTCV dataset
    rows = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 42])

    # Extract each metric from the metrics tensor
    tp = metrics[rows, 0]
    tn = metrics[rows, 1]
    fp = metrics[rows, 2]
    fn = metrics[rows, 3]

    # Calculate accuracy, precision, recall, and F1 score for each class
    accuracy = np.divide(tp + tn, tp + tn + fp + fn, out=np.zeros_like(tp + tn), where=tp + tn + fp + fn!=0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp!=0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=tp + fn!=0)
    f1_score = np.divide(2 * (precision * recall), precision + recall, out=np.zeros_like(2 * (precision * recall)), where=precision + recall!=0)


    print('accuracy')
    print(accuracy)

    print('precision')
    print(precision)

    print('recall')
    print(recall)

    print('F1')
    print(f1_score)

    # Calculate micro metrics
    micro_accuracy = np.sum(tp + tn) / np.sum(tp + tn + fp + fn)
    micro_precision = np.divide(np.sum(tp), np.sum(tp + fp), out=np.zeros_like(np.sum(tp)), where=np.sum(tp + fp)!=0)
    micro_recall = np.divide(np.sum(tp), np.sum(tp + fn), out=np.zeros_like(np.sum(tp)), where=np.sum(tp + fn)!=0)
    micro_f1_score = np.divide(2 * (micro_precision * micro_recall), micro_precision + micro_recall, out=np.zeros_like(2 * (micro_precision * micro_recall)), where=micro_precision + micro_recall!=0)

    # Calculate macro metrics
    macro_accuracy = np.mean(accuracy)
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1_score = np.mean(f1_score)

    median_accuracy = np.median(accuracy)
    median_precision = np.median(precision)
    median_recall = np.median(recall)
    median_f1_score = np.median(f1_score)

    return ((micro_accuracy, micro_precision, micro_recall, micro_f1_score), 
            (macro_accuracy, macro_precision, macro_recall, macro_f1_score),
            (median_accuracy, median_precision, median_recall, median_f1_score))


if __name__ == '__main__':
	folder_path = '../../btcv_data/'

	# get test data
	json_file_path = os.path.join(folder_path, 'data_split.json')
	with open(json_file_path, 'r') as f:
		dataset = json.load(f)
	test_data = dataset['test']

	# define the output metrix: TP, TN, FP, FN
	metrics = np.zeros((105, 4))
	cnt, length = 0, len(test_data)
	for subj in test_data:		
		cnt += 1
		print(f'processing {cnt} / {length} ...')
		# get file name for label and predictions
		label_name = subj['label'].replace('label.nii.gz', 'label_btcv.npy')
		# pred_name = label_name.replace('label', 'pred_label_0629_gaussian')
		pred_name = label_name.replace('label', 'pred_label')

		# get file path for label and predictions
		label_path = os.path.join(folder_path, label_name)
		pred_path = os.path.join(folder_path, pred_name)

		# prepare the pred, label data for calculate
		pred, label = process_subj(pred_path, label_path)

		# calculate the TP, TN, FP, FN for each class
		subj_metrics = count_statistics(pred, label)
		metrics += subj_metrics

	(micro_accuracy, micro_precision, micro_recall, micro_f1_score), (macro_accuracy, macro_precision, macro_recall, macro_f1_score), (median_accuracy, median_precision, median_recall, median_f1_score) = compute_metrics(metrics)

	print("Micro Metrics - Accuracy: {}, Precision: {}, Recall: {}, F1 Score: {}".format(micro_accuracy, micro_precision, micro_recall, micro_f1_score))
	print("Macro Metrics - Accuracy: {}, Precision: {}, Recall: {}, F1 Score: {}".format(macro_accuracy, macro_precision, macro_recall, macro_f1_score))
	print("Median Metrics - Accuracy: {}, Precision: {}, Recall: {}, F1 Score: {}".format(median_accuracy, median_precision, median_recall, median_f1_score))






