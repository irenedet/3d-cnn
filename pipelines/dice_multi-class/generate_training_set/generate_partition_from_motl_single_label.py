from os import makedirs

from file_actions.readers.hdf import _load_hdf_dataset
from src.python.tomogram_utils.random_transformations import transform_data_from_h5

import argparse
from os.path import join
from file_actions.readers.csv import read_motl_from_csv
import numpy as np


def extract_coordinates_from_full_motl(path_to_full_motl: str,
                                       tomo_number: int):
    full_motl = read_motl_from_csv(path_to_full_motl)
    columns_in_tomo = np.where(full_motl[1, :] == tomo_number)[0]
    coordinates = full_motl[2:5, columns_in_tomo]
    angles = full_motl[5:8, columns_in_tomo]
    return np.transpose(coordinates), np.transpose(angles)


def invert_coordinates_entries(coordinates: np.array):
    coordinates_inv = [list(point) for point in coordinates]
    for point in coordinates_inv:
        point.reverse()
    return np.array(coordinates_inv)


def compute_padding(coordinates_zyx, dataset_shape, subtomo_shape) -> list:
    distance_to_right_edges = [dataset_shape - point for point in
                               coordinates_zyx]
    distance_to_left_edges = [point for point in coordinates_zyx]
    right_excess = np.array(
        [subtomo_shape // 2 - distance for distance in distance_to_right_edges])
    left_excess = np.array(
        [subtomo_shape // 2 - distance for distance in distance_to_left_edges])
    thr_right_excess = np.clip(right_excess, a_max=np.max(right_excess),
                               a_min=0)
    thr_left_excess = np.clip(left_excess, a_max=np.max(left_excess), a_min=0)
    right_padding = [np.max(thr_right_excess[:, column]) for column in
                     [0, 1, 2]]
    left_padding = [np.max(thr_left_excess[:, column]) for column in
                    [0, 1, 2]]
    padding = [[int(left_pad), int(right_pad)] for left_pad, right_pad in
               zip(left_padding, right_padding)]
    return padding


from file_actions.writers.h5 import write_joint_raw_and_labels_subtomograms


def partition_raw_and_labels_from_centers(padded_raw_dataset: np.array,
                                          padded_labels_dataset: np.array,
                                          padded_particles_coordinates: list,
                                          label_name: str,
                                          output_h5_file_path: str,
                                          subtomo_shape: tuple):
    write_joint_raw_and_labels_subtomograms(
        output_path=output_h5_file_path,
        padded_raw_dataset=padded_raw_dataset,
        padded_labels_dataset=padded_labels_dataset,
        label_name=label_name,
        window_centers=padded_particles_coordinates,
        crop_shape=subtomo_shape)


parser = argparse.ArgumentParser()
parser.add_argument("-raw", "--path_to_raw",
                    help="path to tomogram to be segmented in hdf format",
                    type=str)
parser.add_argument("-labeled", "--path_to_labeled",
                    help="path to tomogram to with spatial label masks",
                    type=str)
parser.add_argument("-motl", "--path_to_motl",
                    help="path to motls indicating partition centers",
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
                    type=float)
parser.add_argument("-overlap", "--overlap",
                    type=int)
parser.add_argument("-tomo_number", "--tomo_number",
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
overlap = args.overlap
path_to_full_motl = args.path_to_motl  # ToDo use this!
tomo_number = args.tomo_number

coordinates, _ = extract_coordinates_from_full_motl(path_to_full_motl,
                                                    tomo_number)
coordinates_zyx = invert_coordinates_entries(coordinates)
subtomo_shape = box_side * np.array([1, 1, 1])

# output_shape = (shape_y, shape_y, shape_x)
# subtomogram_shape = (box_side, box_side, box_side)
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
raw_dataset = np.array(raw_dataset)
labels_dataset = np.array(labels_dataset)

labels_dataset_shape = np.array(labels_dataset.shape)
raw_dataset_shape = np.array(raw_dataset.shape)

padding_raw = compute_padding(coordinates_zyx, raw_dataset_shape,
                              subtomo_shape)
padding_labels = compute_padding(coordinates_zyx, labels_dataset_shape,
                                 subtomo_shape)

padding = []
for pad_raw, pad_label in zip(padding_raw, padding_labels):
    left_pad_raw, right_pad_raw = pad_raw
    left_pad_label, right_pad_label = pad_label
    left_pad = np.max([left_pad_raw, left_pad_label])
    right_pad = np.max([right_pad_raw, right_pad_label])
    padding += [[left_pad, right_pad]]

print("padding_raw, padding_labels = ", padding_raw, padding_labels)

padded_raw_dataset = np.pad(array=raw_dataset, pad_width=padding,
                            mode="reflect")
padded_labels_dataset = np.pad(array=labels_dataset, pad_width=padding,
                               mode="reflect")

shift_vector = np.array([pad[0] for pad in padding])
padded_particles_coordinates = [point + shift_vector for point in
                                coordinates_zyx]
integer_padded_coordinates = []
for point in padded_particles_coordinates:
    point = [int(p) for p in point]
    integer_padded_coordinates += [point]

# partition_raw_and_labels_from_centers(padded_raw_dataset,
#                                       padded_labels_dataset,
#                                       integer_padded_coordinates,
#                                       label_name,
#                                       output_h5_file_path,
#                                       tuple(subtomo_shape))

print("The training data path is ", output_h5_file_path)
print("split = ", split)

print("Splitting training and testing data into two different files...")
# split_and_write_h5_partition(h5_partition_data_path=output_h5_file_path,
#                              h5_train_patition_path=h5_train_partition_path,
#                              h5_test_patition_path=h5_test_partition_path,
#                              label_name=label_name,
#                              split=split,
#                              shuffle=True)

print("The data augmentation is starting...")
transform_data_from_h5(training_data_path=h5_train_partition_path,
                       label_name=label_name, number_iter=number_iter,
                       output_data_path=output_data_path, split=-1,
                       transform_type='All',
                       sigma_gauss=1,
                       alpha_elastic=5,
                       interp_step=5,
                       p_rotation=0.5,
                       max_angle_rotation=90,
                       only_rotate_xy=True)


print("The training data with data augmentation has been writen in ",
      output_data_path)

print("The script has finished!")
