import argparse
from distutils.util import strtobool
from os import makedirs
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets.actions import \
    generate_strongly_labeled_partition

parser = argparse.ArgumentParser()
parser.add_argument("-segmentation_names", "--segmentation_names",
                    help="segmentation_names",
                    type=str)
parser.add_argument("-output", "--output_dir",
                    help="directory where the outputs will be stored",
                    type=str)
parser.add_argument("-write_on_table", "--write_on_table",
                    help="if True, name of training set will be recorded in db",
                    type=str)
parser.add_argument("-box", "--box_side",
                    type=int)
parser.add_argument("-number_iter", "--number_iter",
                    type=int)
parser.add_argument("-split", "--split",
                    type=float)
parser.add_argument("-dataset_table", "--dataset_table",
                    help="path to db (dataset_table) in .csv format",
                    type=str)
parser.add_argument("-tomo_name", "--tomo_name",
                    help="tomo_name in sessiondate/datanumber format",
                    type=str)
parser.add_argument("-min_label_fraction", "--min_label_fraction",
                    help="min_label_fraction",
                    type=float)
parser.add_argument("-max_label_fraction", "--max_label_fraction",
                    help="max_label_fraction",
                    type=float)

args = parser.parse_args()
tomo_name = args.tomo_name
dataset_table = args.dataset_table
output_dir = args.output_dir
min_label_fraction = args.min_label_fraction
max_label_fraction = args.max_label_fraction
box_side = args.box_side
number_iter = args.number_iter
split = args.split
segmentation_names = args.segmentation_names
write_on_table = strtobool(args.write_on_table)

segmentation_names = list(map(str, segmentation_names.split(',')))
print(segmentation_names)
overlap = 12
print("output_dir", output_dir)

df = pd.read_csv(dataset_table)
df['tomo_name'] = df['tomo_name'].astype(str)
tomo_df = df[df['tomo_name'] == tomo_name]
path_to_raw = tomo_df.iloc[0]['eman2_filetered_tomo']

labels_dataset_list = list()
for semantic_class in segmentation_names:
    mask_name = semantic_class + '_mask'
    path_to_mask = tomo_df.iloc[0][mask_name]
    labels_dataset_list.append(path_to_mask)

print("labels_dataset_list = ")
print(labels_dataset_list)

subtomogram_shape = (box_side, box_side, box_side)
# output_dir = join(output_dir, "train_and_test_partitions")
output_h5_file_name = "full_partition.h5"
output_path = join(output_dir, output_h5_file_name)
makedirs(name=output_dir, exist_ok=True)

label_fractions_list = generate_strongly_labeled_partition(
    path_to_raw=path_to_raw,
    labels_dataset_paths_list=labels_dataset_list,
    segmentation_names=segmentation_names,
    output_h5_file_path=output_path,
    subtomo_shape=subtomogram_shape,
    overlap=overlap,
    min_label_fraction=min_label_fraction,
    max_label_fraction=max_label_fraction)

fig_path = join(output_dir, "label_fractions_hist.png")
print("The training data path is ", output_path)
zoom = np.where(np.array(label_fractions_list) < 0.01)[0]
plt.figure(1)
plt.hist(label_fractions_list, bins=10)
plt.xlabel("label fraction")
plt.ylabel("frequency")
plt.gcf()
plt.savefig(fname=fig_path, format="png")

fig_path = join(output_dir, "zoom_label_fractions_hist.png")
plt.figure(2)
plt.hist(np.array(label_fractions_list)[zoom], bins=10)
plt.xlabel("label_fraction < 0.01")
plt.ylabel("frequency")
plt.gcf()
plt.savefig(fname=fig_path, format="png")

if write_on_table:
    print("path to training partition written on table: ", output_path)
    df.loc[
        df['tomo_name'] == tomo_name, 'train_partition'] = output_path
    df.to_csv(path_or_buf=dataset_table, index=False)
