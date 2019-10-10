from os import makedirs
from os.path import join
import pandas as pd
from distutils.util import strtobool
from src.python.datasets.random_transformations import \
    transform_data_from_h5_dice_multi_class
from src.python.datasets.actions import \
    partition_raw_and_labels_tomograms_dice_multiclass, \
    generate_strongly_labeled_partition
from src.python.filewriters.h5 import \
    split_and_write_h5_partition_dice_multi_class

import argparse

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

args = parser.parse_args()
tomo_name = args.tomo_name
dataset_table = args.dataset_table
output_dir = args.output_dir
min_label_fraction = args.min_label_fraction
box_side = args.box_side
number_iter = args.number_iter
split = args.split
segmentation_names = args.segmentation_names
write_on_table = strtobool(args.write_on_table)

# dataset_table = "/struct/mahamid/Irene/liang_data/liang_data.csv"
# global_output_dir = "/scratch/trueba/liang_data"
# write_on_table = 'True'
# segmentation_names = 'ribo'  # ,fas,memb'
# split = 0.7
# box_side = 128
# number_iter = 1
# tomo_name = "172"
# output_dir = global_output_dir + "/" + tomo_name
# segmentation_names = "ribo"

segmentation_names = list(map(str, segmentation_names.split(',')))
print(segmentation_names)
overlap = 12
print("output_dir", output_dir)

df = pd.read_csv(dataset_table)
df['tomo_name'] = df['tomo_name'].astype(str)
tomo_df = df[df['tomo_name'] == tomo_name]
x_dim = int(tomo_df.iloc[0]['x_dim'])
y_dim = int(tomo_df.iloc[0]['y_dim'])
z_dim = int(tomo_df.iloc[0]['z_dim'])
path_to_raw = tomo_df.iloc[0]['eman2_filetered_tomo']

labels_dataset_list = list()
for semantic_class in segmentation_names:
    mask_name = semantic_class + '_mask'
    path_to_mask = tomo_df.iloc[0][mask_name]
    labels_dataset_list.append(path_to_mask)

print("labels_dataset_list = ")
print(labels_dataset_list)

output_shape = (z_dim, y_dim, x_dim)
subtomogram_shape = (box_side, box_side, box_side)
output_dir = join(output_dir, "train_and_test_partitions")
output_h5_file_name = "full_partition.h5"
output_h5_file_path = join(output_dir, output_h5_file_name)
output_data_path = join(output_dir, "data_aug_on_train_partition.h5")

####################
# For splitting test and train sets:
h5_train_partition_path = join(output_dir, "train_partition.h5")
h5_test_partition_path = join(output_dir, "test_partition.h5")

#####################

makedirs(name=output_dir, exist_ok=True)

# partition_raw_and_labels_tomograms_dice_multiclass(
#     path_to_raw=path_to_raw,
#     labels_dataset_list=labels_dataset_list,
#     segmentation_names=segmentation_names,
#     output_h5_file_path=output_h5_file_path,
#     subtomo_shape=subtomogram_shape,
#     overlap=overlap)

generate_strongly_labeled_partition(path_to_raw=path_to_raw,
                                    labels_dataset_list=labels_dataset_list,
                                    segmentation_names=segmentation_names,
                                    output_h5_file_path=output_h5_file_path,
                                    subtomo_shape=subtomogram_shape,
                                    overlap=overlap,
                                    min_label_fraction=min_label_fraction)
print("The training data path is ", output_h5_file_path)
#
# print("Splitting training and testing data into two different files...")
# split_and_write_h5_partition_dice_multi_class(
#     h5_partition_data_path=output_h5_file_path,
#     h5_train_patition_path=h5_train_partition_path,
#     h5_test_patition_path=h5_test_partition_path,
#     segmentation_names=segmentation_names,
#     split=split,
#     shuffle=True)
# print("The training set has been written in ", h5_train_partition_path)
# print("The testing set has been written in ", h5_test_partition_path)

# print("The data augmentation is starting...")
# transform_data_from_h5_dice_multi_class(
#     training_data_path=h5_train_partition_path,
#     segmentation_names=segmentation_names,
#     number_iter=number_iter,
#     output_data_path=output_data_path)
# print("The training data with data augmentation has been writen in ",
#       output_data_path)

# print("The script has finished!")

if write_on_table:
    print("path to training partition written on table: ", output_h5_file_path)
    df.loc[
        df['tomo_name'] == tomo_name, 'train_partition'] = output_h5_file_path
    df.to_csv(path_or_buf=dataset_table, index=False)
