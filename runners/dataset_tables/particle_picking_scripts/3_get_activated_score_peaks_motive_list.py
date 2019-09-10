import pandas as pd
import numpy as np
import argparse
from os import makedirs
from os.path import join
import torch.nn as nn

from src.python.filewriters.csv import \
    write_global_motl_from_overlapping_subtomograms_multiclass

parser = argparse.ArgumentParser()

parser.add_argument("-output", "--output_dir",
                    help="directory where the outputs will be stored",
                    type=str)
parser.add_argument("-label", "--label_name",
                    help="name of category to be segmented",
                    type=str)
parser.add_argument("-dataset_table", "--dataset_table",
                    help="path to dataset_table",
                    type=str)
parser.add_argument("-tomo_name", "--tomo_name",
                    help="tomo to be analyzed",
                    type=str)
parser.add_argument("-box", "--box_side",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-overlap", "--overlap",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-class_number", "--class_number",
                    help="class number associated to motifs list",
                    type=int)
parser.add_argument("-min_peak_distance", "--min_peak_distance",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-final_activation", "--final_activation",
                    help="name of final_activation to apply to segmentation",
                    type=str, default="elu")

args = parser.parse_args()
output_dir = args.output_dir
label_name = args.label_name
dataset_table = args.dataset_table
tomo_name = args.tomo_name
box_side = args.box_side
overlap = args.overlap
class_number = args.class_number
min_peak_distance = args.min_peak_distance
final_activation = args.final_activation

if final_activation == "elu":
    final_activation = nn.ELU()
else:
    final_activation = None


df = pd.read_csv(dataset_table)
df['tomo_name'] = df['tomo_name'].astype(str)
tomo_df = df[df['tomo_name'] == tomo_name]
z_shift = int(tomo_df.iloc[0]['z_shift'])
x_dim = int(tomo_df.iloc[0]['x_dim'])
y_dim = int(tomo_df.iloc[0]['y_dim'])
z_dim = int(tomo_df.iloc[0]['z_dim'])
subtomo_path = tomo_df.iloc[0]['test_partition']
print(subtomo_path)

subtomo_shape = tuple(box_side * np.array([1, 1, 1]))
output_shape = (z_dim, y_dim, x_dim)

peaks_per_subtomo = int(box_side ** 3 / (2 * min_peak_distance) ** 3)
print("peaks per subtomo = ", peaks_per_subtomo)
number_peaks_uniquify = 7000
output_dir = join(output_dir, "peaks")
makedirs(name=output_dir, exist_ok=True)


motl_file_name = write_global_motl_from_overlapping_subtomograms_multiclass(
    subtomograms_path=subtomo_path,
    motive_list_output_dir=output_dir,
    overlap=overlap,
    label_name=label_name,
    output_shape=output_shape,
    subtomo_shape=subtomo_shape,
    numb_peaks=peaks_per_subtomo,
    class_number=class_number,
    min_peak_distance=min_peak_distance,
    number_peaks_uniquify=number_peaks_uniquify,
    z_shift=z_shift,
    final_activation=final_activation)

print(motl_file_name)
