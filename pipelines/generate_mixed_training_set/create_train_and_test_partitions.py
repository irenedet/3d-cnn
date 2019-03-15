# from os.path import join
from os import makedirs

from src.python.filereaders.hdf import _load_hdf_dataset
from src.python.datasets.transformations import transform_data_from_h5
from src.python.datasets.actions import partition_raw_and_labels_tomograms
from src.python.filewriters.h5 import split_and_write_h5_partition

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-raw", "--path_to_raw",
                    help="path to tomogram to be segmented in hdf format",
                    type=str)
parser.add_argument("-labeled", "--path_to_labeled",
                    help="path to tomogram to with spatial label masks",
                    type=str)
parser.add_argument("-output", "--output_dir",
                    help="directory where the outputs will be stored",
                    type=str)
parser.add_argument("-label", "--label_name",
                    type=str)
parser.add_argument("-box", "--box_side",
                    type=int)
parser.add_argument("-shapex", "--output_shape_x",
                    type=int)
parser.add_argument("-shapey", "--output_shape_y",
                    type=int)
parser.add_argument("-shapez", "--output_shape_z",
                    type=int)
parser.add_argument("-number_iter", "--number_iter",
                    type=int)
parser.add_argument("-split", "--split",
                    type=int)
parser.add_argument("-train_split", "--train_split",
                    type=int)
parser.add_argument("-overlap", "--overlap",
                    type=int)

args = parser.parse_args()
path_to_raw = args.path_to_raw
path_to_labeled = args.path_to_labeled
output_dir = args.output_dir
label_name = args.label_name
shape_x = args.output_shape_x
shape_y = args.output_shape_y
shape_z = args.output_shape_z
box_side = args.box_side
number_iter = args.number_iter
split = args.split
train_split = args.train_split
overlap = args.overlap

from os.path import join

# path_to_raw = "/scratch/trueba/cnn/004/4bin/cnn/rawtomogram/180426_004_4bin.hdf"
# path_to_labeled = "/scratch/trueba/cnn/004/4bin/cnn/centralregion_004.hdf"
# output_dir = "/scratch/trueba/3d-cnn/training_data/TEST/004_last"
# label_name = "ribosomes"
# split = 130  # Between partitions of testing and training data
# shape_x = 928
# shape_y = 928
# shape_z = 221
# box_side = 128
#
# # For data augmentation:
# number_iter = 6
# train_split = 110  # within training data for nnet training
# overlap = 12

assert split > train_split

output_shape = (shape_y, shape_y, shape_x)
subtomogram_shape = (box_side, box_side, box_side)
output_dir = join(output_dir, "train_and_test_partitions")
output_h5_file_name = "partition_training.h5"
output_h5_file_path = join(output_dir, output_h5_file_name)
output_data_path = join(output_dir, "data_aug_on_training_split.h5")

####################
# For splitting test and train sets:
h5_train_partition_path = join(output_dir, "train_partition.h5")
h5_test_partition_path = join(output_dir, "test_partition.h5")

#####################

makedirs(name=output_dir, exist_ok=True)

raw_dataset = _load_hdf_dataset(hdf_file_path=path_to_raw)
labels_dataset = _load_hdf_dataset(hdf_file_path=path_to_labeled)
partition_raw_and_labels_tomograms(raw_dataset=raw_dataset,
                                   labels_dataset=labels_dataset,
                                   label_name=label_name,
                                   output_h5_file_path=output_h5_file_path,
                                   subtomo_shape=subtomogram_shape,
                                   overlap=overlap)
print("The training data path is ", output_h5_file_path)

print("Splitting training and testing data into two different files...")
split_and_write_h5_partition(h5_partition_data_path=output_h5_file_path,
                             h5_train_patition_path=h5_train_partition_path,
                             h5_test_patition_path=h5_test_partition_path,
                             split=split)
print("The training set has been written in ", h5_train_partition_path)
print("The testing set has been written in ", h5_test_partition_path)

print("The data augmentation is starting...")
transform_data_from_h5(training_data_path=h5_train_partition_path,
                       label_name=label_name, number_iter=number_iter,
                       output_data_path=output_data_path, split=train_split)
print("The training data with data augmentation has been writen in ",
      output_data_path)

print("The script has finished!")
