#!/usr/bin/python
# -*- coding: UTF-8 -*-

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap


def export_color_list(rgb_values, labels, index_val, template_path, save_path):
	# Assuming additional fields are filled with 1s, and are of the same length as labels and RGB values
	additional_fields = np.ones((len(labels), 3))

	# Prepare final array
	# final_array = np.hstack((np.arange(0, len(labels))[:, None].astype(int), rgb_values.astype(int), additional_fields.astype(int), np.array(labels)[:, None]))
	final_array = np.hstack((np.array(index_val)[:, None].astype(int), rgb_values.astype(int), additional_fields.astype(int), np.array(labels)[:, None]))

	# find the index of background and change it to be zeros
	idx = labels.index('background')
	final_array[idx, :-1] = 0

	# sort it by the index val
	sorted_indices = np.argsort(np.array(index_val).reshape((len(index_val), 1))[:, 0])
	sorted_array = final_array[sorted_indices]

	# get header information from template
	with open(template_path, 'r') as template_f:
		header = template_f.read()

	# Open a file in write mode
	with open(save_path, 'w') as f:
		# Write formatted header to file
		f.write(header)

		# Loop through the data and write each row to file
		for item in sorted_array:
			length = len(item[7])
			f.write(f"{item[0]:>5} {item[1]:>5} {item[2]:>4} {item[3]:>4} {item[4]:>8} {item[5]:>2} {item[6]:>2}    \"{item[7]:<{length}}\"\n")


def create_color_map(n):
	''' create n different RGB colors for visualization
	'''
	base = plt.cm.get_cmap('nipy_spectral', n)
	color_list = base(np.linspace(0.1, 1, n))
	cmap_name = base.name + str(n)
	color_list_rgb = (color_list * 255).astype(int)
	return color_list_rgb


def load_label(json_file_path):
	''' read the label information from json file
	input: json file path
	output: labels and corresponding values
	'''
	# load the json file
	with open(json_file_path, 'r') as f:
		data = json.load(f)

	# the key is the label name, and value is the assigned label value
	keys = list(data.keys())
	vals = list(data.values())
	vals = list(map(int, vals))

	# make sure background information is correct
	if 'background' in keys and 0 in vals:
		key_idx = keys.index('background')
		val_idx = vals.index(0)
		if key_idx != val_idx:
			raise ValueError('The background label is not associated with 0')
	elif 'background' not in keys and 0 not in vals:
		keys = ['background'] + keys
		vals = [0] + vals
	elif 'background' not in keys and 0 in vals:
		raise ValueError('background is not included, and index 0 has been assigned to other labels')
	elif 'background' in keys and 0 not in vals:
		raise ValueError('background label is associated with other index other than 0')
	else:
		pass
	return keys, vals


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Process some files.')
    
	parser.add_argument('--label_path', type=str, default='./totalSegmentator_label.json', help='Path to JSON file')
	parser.add_argument('--template_path', type=str, default='./template.txt', help='Path to template file')
	parser.add_argument('--save_path', type=str, default='./totalSegmentator_color_itksnap.txt', help='Path to save output file')
    
	args = parser.parse_args()
	
	# get label name from json file, eg {'background': 0, 'spleen': 1, 'kidney_left': 2}
	label_name, index_val = load_label(args.label_path)

	# create color map
	num_label = len(label_name)
	color_list_rgb = create_color_map(num_label)

	# assign the colors to labels, and save as txt file for itksnap visualization
	export_color_list(color_list_rgb[:, :3], label_name, index_val, args.template_path, args.save_path)

