# Data Splitting Script

## Description

This Python script allows for data splitting into train, dev, and test sets. It supports both single-fold and n-fold splitting methods.

## Requirements

- **Python 3.x**
- **NumPy**
- **sklearn**
- **json**
- **argparse**
- **os**

## Installation

You can install the required packages using `pip`:

```bash
pip install numpy scikit-learn
```

## Usage

### Single-Fold Split

Performs a single-fold data split based on custom or default ratios for training, development, and testing datasets.

#### Syntax:

```bash
python data_split.py --data_path your_custom_data_path --n_folds 1 --save_folder './'
```

#### With custom ratios:

```bash
python data_split.py --data_path your_custom_data_path --train_ratio 0.7 --dev_ratio 0.2 --test_ratio 0.1 --n_folds 1 --save_folder './'
```

### N-Fold Split

Performs n-fold data splitting, generating separate JSON files for each fold.

#### Syntax:

```bash
python data_split.py --data_path your_custom_data_path --n_folds 5 --save_folder './'
```

### Optional: Custom Data Path

For both single-fold and n-fold, if you wish to specify a custom path for the data, use:

```bash
python data_split.py --data_path your_custom_data_path --n_folds 1 --save_folder './'  # For single-fold
```

or

```bash
python data_split.py --data_path your_custom_data_path --n_folds 5 --save_folder './'  # For n-fold
```

Once the data_split.json file is generated, move it to the data folder for later process.