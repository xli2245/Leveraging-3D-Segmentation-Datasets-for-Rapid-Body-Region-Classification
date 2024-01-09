import os
import argparse
import json
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt


def load_and_process_nifti(img_path, label_path):
    # Load the image
    img = nib.load(img_path)
    label_file = nib.load(label_path)

    # Get the data as a numpy array
    data = img.get_fdata()
    label = label_file.get_fdata()

    # Step 1: Clip values
    minimum, maximum = -1000, 2000
    data = np.clip(data, minimum, maximum)

    # Step 2: Padding
    x, y, z = data.shape
    padded_size = 64
    # if squeeze through the coronal view
    if x < padded_size and z < padded_size:
        padding_x = (padded_size - x) // 2
        padding_z = (padded_size - z) // 2
        padding = ((padding_x, padded_size - x - padding_x), (0, 0), (padding_z, padded_size - z - padding_z))
    elif x < padded_size and z >= padded_size:
        padding_x = (padded_size - x) // 2
        padding = ((padding_x, padded_size - x - padding_x), (0, 0), (0, 0))
    elif x >= padded_size and z < padded_size:
        padding_z = (padded_size - z) // 2
        padding = ((0, 0), (0, 0), (padding_z, padded_size - z - padding_z))
    else:
        padding = ((0, 0), (0, 0), (0, 0))
    data = np.pad(data, padding, 'constant', constant_values=minimum)
    label = np.pad(label, padding, 'constant', constant_values=0)
    label_x, label_y, label_z = label.shape

    # Step 3: deal with labels to make it to be multi-hot vector
    new_label = np.zeros((label_x, label_z, 105))
    for i in range(label_x):
        for j in range(label_z):
            unique_values = np.unique(label[i, :, j])
            new_label[i, j, unique_values.astype(int)] = 1

    # Step 4: data Normalization
    data = (data - minimum) / (maximum - minimum)

    # Step 5: Squeeze to 2D by averaging along the y dimension
    data = np.mean(data, axis=1)

    # rotation to make the image look exactly as what is shown in the crononal view.
    # countclockwise rotate 90 degrees
    rotated_data = np.rot90(data, k=1, axes=(0, 1))
    rotated_label = np.rot90(new_label, k=1, axes=(0, 1))
    # horizontal flipping
    final_data = np.fliplr(rotated_data)
    final_label = np.fliplr(rotated_label)

    return final_data, final_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../totalseg_data', help='data path')
    parser.add_argument('--data_split_name', type=str, default='data_split.json', help='name of the data split file')
    parser.add_argument('--ori_image_name', type=str, default='ct.nii.gz', help='original image name')
    parser.add_argument('--ori_label_name', type=str, default='label.nii.gz', help='original label name')
    parser.add_argument('--target_image_name', type=str, default='processed_ct.npy', help='processed image name')
    parser.add_argument('--target_label_name', type=str, default='processed_label.npy', help='processed label name')
    args = parser.parse_args()

    folders = args.data_path
    json_path = os.path.join(folders, args.data_split_name)
    ori_img_name = args.ori_image_name
    ori_label_name = args.ori_label_name
    target_img_name = args.target_image_name
    target_label_name = args.target_label_name

    # open original json file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Replace the specified strings
    for split in data:
        for item in data[split]:
            # get the image and label nifti file path
            img_path = os.path.join(folders, item['image'])
            label_path = os.path.join(folders, item['label'])
            
            # img_path = '../../totalseg_data/s0579/ct.nii.gz'
            # label_path = '../../totalseg_data/s0579/label.nii.gz'

            # load and process the image and label nifti files
            squeezed_img, squeezed_label = load_and_process_nifti(img_path, label_path)

            # save the processed image into npy files
            saved_img_path = img_path.replace(ori_img_name, target_img_name)
            saved_label_path = label_path.replace(ori_label_name, target_label_name)

            squeezed_label = squeezed_label.astype(np.uint8)

            np.save(saved_img_path, squeezed_img)
            np.save(saved_label_path, squeezed_label)

            # save the png files for visualization
            plt.imshow(squeezed_img * 255, cmap="gray")
            plt.savefig(saved_img_path.replace('.npy', '.png'))

            # update the image and label path to the path of processed image and label
            item['image'] = item['image'].replace(ori_img_name, target_img_name)
            item['label'] = item['label'].replace(ori_label_name, target_label_name)


    new_json_path = json_path.replace('.json', '_processed.json')
    # Save the modified JSON back to the file
    with open(new_json_path, 'w') as f:
        json.dump(data, f, indent=2)

