# Color Exporter for ITK-SNAP use

This script takes a JSON file containing label names and their respective index values, generates a colormap for these labels, and saves it as a `.txt` file suitable for use with ITK-SNAP. This allows you to visualize different labeled regions in medical images with distinct colors.

## Requirements

- **Python 3.x**
- **NumPy**
- **Matplotlib**

## Installation

You can install the required packages using `pip`:

```bash
pip install numpy matplotlib
```

## Usage

### Command-line Arguments

The script accepts the following optional command-line arguments:

--label_path: Path to the JSON file containing the labels and index values.
Default: ./totalSegmentator_label.json

--template_path: Path to the template.txt file that contains some necessary information.
Default: ./template.txt

--save_path: Path where the output txt file will be saved.
Default: ./totalSegmentator_color_itksnap.txt


### Example Command

Run the script with the following command:

```bash
python your_script_name.py --label_path "./custom_label.json" --save_path "./custom_color_itksnap.txt"
```

