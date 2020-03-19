import argparse
from distutils.util import strtobool
from os import makedirs
from os.path import join

import numpy as np
import pandas as pd

from constants.dataset_tables import DatasetTableHeader
from constants.partitions import RANDOM_PARTITION_NAME
from tomogram_utils.volume_actions.actions import \
    generate_random_labeled_partition

parser = argparse.ArgumentParser()
parser.add_argument("-segmentation_names", "--segmentation_names",
                    help="segmentation_names",
                    type=str)
parser.add_argument("-output", "--output_dir",
                    help="directory where the output will be stored",
                    type=str)
parser.add_argument("-write_on_table", "--write_on_table",
                    help="if True, name of training set will be recorded in "
                         "the dataset table",
                    type=str)
parser.add_argument("-box", "--box_side",
                    type=int)
parser.add_argument("-n_total", "--n_total",
                    type=int)
parser.add_argument("-dataset_table", "--dataset_table",
                    help="path to db (dataset_table) in .csv format",
                    type=str)
parser.add_argument("-partition_name", "--partition_name",
                    help="partition_name column where partition will be stored "
                         "in the dataset table",
                    type=str, default="train_partition")
parser.add_argument("-processing_tomo", "--processing_tomo",
                    help="name of processing_tomo in dataset_table"
                         "that will be partitioned",
                    type=str, default='eman2_filtered_tomo')
parser.add_argument("-image_acquisition_parameter",
                    "--image_acquisition_parameter",
                    help="image_acquisition_parameter such as defocus/vpp",
                    type=str, default='vpp')
parser.add_argument("-tomo_name", "--tomo_name",
                    help="tomo_name",
                    type=str)
parser.add_argument("-min_label_fraction", "--min_label_fraction",
                    help="min_label_fraction",
                    type=float)
parser.add_argument("-max_label_fraction", "--max_label_fraction",
                    help="max_label_fraction",
                    type=float)
parser.add_argument("-prefix", "--prefix",
                    help="additional name reference for the set to partition",
                    type=str or int, default="false")
parser.add_argument("-edge_tolerance", "--edge_tolerance",
                    help="edge_tolerance to not include particles half padded",
                    type=int, default=0)

args = parser.parse_args()
partition_name = args.partition_name
tomo_name = args.tomo_name
dataset_table = args.dataset_table
output_dir = args.output_dir
min_label_fraction = args.min_label_fraction
max_label_fraction = args.max_label_fraction
box_side = args.box_side
segmentation_names = args.segmentation_names
write_on_table = strtobool(args.write_on_table)
prefix = args.prefix
edge_tolerance = args.edge_tolerance
processing_tomo = args.processing_tomo
image_parameter = args.image_acquisition_parameter
segmentation_names = list(map(str, segmentation_names.split(',')))
print(segmentation_names)
n_total = args.n_total
print("output_dir", output_dir)
DTHeader = DatasetTableHeader(semantic_classes=segmentation_names,
                              processing_tomo=processing_tomo,
                              image_acquisition_parameter=image_parameter,
                              partition_name=partition_name)

df = pd.read_csv(dataset_table)
df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)

if prefix == "false":
    tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
else:
    tomo_df = df[
        (df[DTHeader.tomo_name] == tomo_name) & (df["subset_prefix"] == prefix)]

path_to_raw = tomo_df.iloc[0][DTHeader.processing_tomo]
labels_dataset_list = list()
for mask_name in DTHeader.masks_names:
    path_to_mask = tomo_df.iloc[0][mask_name]
    labels_dataset_list.append(path_to_mask)

print("labels_dataset_list = ")
print(labels_dataset_list)

subtomogram_shape = (box_side, box_side, box_side)
output_h5_file_name = RANDOM_PARTITION_NAME
output_path = join(output_dir, output_h5_file_name)
makedirs(output_dir, exist_ok=True)

label_fractions_list = generate_random_labeled_partition(
    path_to_raw=path_to_raw,
    labels_dataset_paths_list=labels_dataset_list,
    segmentation_names=segmentation_names,
    output_h5_file_path=output_path,
    subtomo_shape=subtomogram_shape,
    n_total=n_total,
    min_label_fraction=min_label_fraction,
    max_label_fraction=max_label_fraction,
    edge_tolerance=edge_tolerance)

# print("label_fractions_list =", label_fractions_list)
print(len(np.where(np.array(label_fractions_list) > 0)[0]), "out of",
      len(label_fractions_list), " cubes selected in partition file.")

if write_on_table:
    print("path to training partition written on table: ", output_path)
    df.loc[
        df[DTHeader.tomo_name] == tomo_name, DTHeader.partition_name] = \
        output_path
    df.to_csv(path_or_buf=dataset_table, index=False)
