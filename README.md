# Leveraging 3D Segmentation Datasets for Rapid Body Region Classification
This study aims at efficient 2D body region classification with position information by leveraging 3D segmentation dataset. Here is an example patient from the test dataset. From left to right are the Reference 3D Segmentation and 2D Flattened CT image, in the coronal view, with a visualization of the 2D classification reference and model’s prediction. In the 2D classification visualization, the horizontal axis represents classes, denoted by different colors, while the vertical axis corresponds to the slice location.
![Example result](https://github.com/xli2245/Leveraging-3D-Segmentation-Datasets-for-Rapid-Body-Region-Classification/blob/main/figures/result_visualization.png)
## Table of Contents
- [Setup](#setup)
- [Environment](#environment)
- [Color Exporter for ITK-SNAP use](#color-exporter-for-itk-snap-use)
- [Data preparation](#data-preparation)
  - [Dataset](#dataset)
  - [Data split](#data-split)
- [Training and Evaluation](#training-and-evaluation)
  - [Model training](#model-training)
  - [Model prediction](#model-prediction)
  - [Performance evaluation](#performance-evaluation)
- [Results](#results)
## Setup
Clone this repo:
```
git clone https://github.com/xli2245/Leveraging-3D-Segmentation-Datasets-for-Rapid-Body-Region-Classification
```
## Environment
This study is performed within the [MONAI Docker](https://hub.docker.com/r/projectmonai/monai)

## Color Exporter for ITK-SNAP use
The tool takes a JSON file containing label names and their respective index values, generates a colormap for these labels, and saves it as a `.txt` file suitable for use with ITK-SNAP. This allows you to visualize different labeled regions in medical images with distinct colors. Here is an example usage.
```
python ./1-text_file_generation/color_configuration_for_itknap_use.py --label_path "./1-text_file_generation/custom_label.json" --save_path "./1-text_file_generation/custom_color_itksnap.txt"
```
## Data preparation
### Dataset
Dataset is downloaded from [the TotalSegmentator CT dataset](https://academictorrents.com/details/337819f0e83a1c1ac1b7262385609dad5d485abf) and extracted to the data folder.
### Data split
The downloaded dataset can be split by a self-defined ratio and into certain folds for training.
#### Single-Fold Split
Performs a single-fold data split based on custom or default ratios for training, development, and testing datasets w/o customized ratio.
```bash
python data_split.py --data_path your_custom_data_path --n_folds 1 --save_folder './'
```
```bash
python data_split.py --data_path your_custom_data_path --train_ratio 0.7 --dev_ratio 0.2 --test_ratio 0.1 --n_folds 1 --save_folder './'
```
#### N-Fold Split
Performs n-fold data splitting, generating separate JSON files for each fold.
```bash
python data_split.py --data_path your_custom_data_path --n_folds 5 --save_folder './'
```

## Training and Evaluation
### Model training
```
python ./3-train_and_predict/train.py
```
Here is the training process.
![Training process](https://github.com/xli2245/Leveraging-3D-Segmentation-Datasets-for-Rapid-Body-Region-Classification/blob/main/figures/training.png)

### Model prediction
```
python ./3-train_and_predict/predict.py
```
### Performance evaluation
The accuracy, precision, recall, and F1 score are calculated at micro- and macro- level, respectively.
```
python ./4-evaluation/evaluation_subj_based_btcv.py
```
## Results
1. Running time and memory usage
    ~1.4s averaged running time per patient using < 600MB RAM on a commodity GPU (NVIDIA 1080Ti) while ~53s with ~3000MB RAM for the TotalSegmentator tool with size (249, 188, 213)

2. Model evaluation on the TotalSegmentator test dataset (243 patients)
    Micro-level: accuracy (83.67%), precision (85.12%), recall (87.71%), F1 score (86.40%)
    Macro-level: accuracy (83.01%), precision (86.33%), recall (86.08%), F1 score (84.95%)
    
3. Model evaluation on the BTCV dataset (13 classes, 30 patients)
    Micro-level: accuracy (90.00%), precision (99.43%), recall (90.46%), F1 score (94.74%)
    Macro-level: accuracy (90.00%), precision (99.47%), recall (90.49%), F1 score (94.03%)

