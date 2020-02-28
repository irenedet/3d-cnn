# from os.path import join
from os import makedirs

from file_actions.readers.hdf import _load_hdf_dataset
from src.python.tomogram_utils.random_transformations import transform_data_from_h5

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
parser.add_argument("-outh5", "--output_data_path",
                    help="file where the training set will be stored",
                    type=str)
parser.add_argument("-outh5_aug", "--output_data_path_aug",
                    help="where the training set + data augmentation is stored",
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
parser.add_argument("-overlap", "--overlap",
                    type=int)

args = parser.parse_args()
path_to_raw = args.path_to_raw
path_to_labeled = args.path_to_labeled
output_dir = args.output_dir
label_name = args.label_name
output_h5_file_path = args.output_data_path
output_data_path = args.output_data_path_aug
shape_x = args.output_shape_x
shape_y = args.output_shape_y
shape_z = args.output_shape_z
subtomogram_size = args.box_side
box_side = args.box_side
number_iter = args.number_iter
split = args.split
overlap = args.overlap


output_shape = (shape_z, shape_y, shape_x)
subtomogram_shape = (box_side, box_side, box_side)

from os.path import join
path_to_raw = "/scratch/trueba/cnn/004/4bin/cnn/rawtomogram/180426_004_4bin.hdf"
path_to_labeled = "/scratch/trueba/cnn/004/4bin/cnn/centralregion_004.hdf"

output_dir = "/scratch/trueba/3d-cnn/training_data/TEST/004_last/"


label_name = "ribosomes"
output_h5_file_name = "ribo_training.h5"
output_h5_file_path = join(output_dir, output_h5_file_name)

output_shape = (221, 928, 928)
subtomogram_shape = (128, 128, 128)
#
#
# For data augmentation:
number_iter = 6
split = 130  # Only augment training data

output_data_path = join(output_dir, "data_aug_004_iter10_split130_more_noise.h5")
overlap = 12
#####################

makedirs(name=output_dir, exist_ok=True)

raw_dataset = _load_hdf_dataset(hdf_file_path=path_to_raw)
labels_dataset = _load_hdf_dataset(hdf_file_path=path_to_labeled)
print("raw.shape", raw_dataset.shape)
print("labels_dataset.shape", labels_dataset.shape)
# partition_raw_and_labels_tomograms(raw_dataset=raw_dataset,
#                                    labels_dataset=labels_dataset,
#                                    label_name=label_name,
#                                    output_h5_file_path=output_h5_file_path,
#                                    subtomo_shape=subtomogram_shape,
#                                    overlap=overlap)

print("The training data path is ", output_h5_file_path)

print("The data augmentation is starting...")

transform_data_from_h5(training_data_path=output_h5_file_path,
                       label_name=label_name, number_iter=number_iter,
                       output_data_path=output_data_path, split=split)
print("The training data with data augmentation has been writen in ",
      output_data_path)
print("The script has finished!")
