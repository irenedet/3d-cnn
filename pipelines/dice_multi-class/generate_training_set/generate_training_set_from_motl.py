from os import makedirs

from src.python.filereaders.datasets import load_dataset
from src.python.filewriters.h5 import write_joint_raw_and_labels_subtomograms
from src.python.filewriters.h5 import write_classification_dataset

import argparse
from os.path import join
from src.python.filereaders.csv import read_motl_from_csv
import numpy as np
from src.python.peak_toolbox.utils import read_motl_coordinates_and_values


def extract_coordinates_from_full_motl(path_to_full_motl: str,
                                       tomo_number: int):
    full_motl = read_motl_from_csv(path_to_full_motl)
    columns_in_tomo = np.where(full_motl[1, :] == tomo_number)[0]
    coordinates = full_motl[2:5, columns_in_tomo]
    angles = full_motl[5:8, columns_in_tomo]
    return np.transpose(coordinates), np.transpose(angles)


def invert_coordinates_entries(coordinates: np.array):
    coordinates_inv = [list(point).reverse() for point in coordinates]
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


def generate_classification_file(raw_dataset: np.array,
                                 center_coordinates: list,
                                 label_name: str,
                                 h5_output_path: str,
                                 subtomo_shape: tuple or list):
    write_classification_dataset(
        output_path=h5_output_path,
        padded_raw_dataset=raw_dataset,
        label_name=label_name,
        window_centers=center_coordinates,
        crop_shape=subtomo_shape)


parser = argparse.ArgumentParser()
parser.add_argument("-raw", "--path_to_raw",
                    help="path to tomogram to be segmented in hdf format",
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
path_to_motl = args.path_to_motl
tomo_number = args.tomo_number

values, coordinates = read_motl_coordinates_and_values(path_to_motl)

coordinates_zyx = invert_coordinates_entries(coordinates)
subtomo_shape = box_side * np.array([1, 1, 1])
output_dir = join(output_dir, "classification_set")

output_h5_file_name = "full_classification_dataset.h5"
output_h5_file_path = join(output_dir, output_h5_file_name)
h5_train_partition_path = join(output_dir, "training_set.h5")
h5_test_partition_path = join(output_dir, "testing_set.h5")
makedirs(name=output_dir, exist_ok=True)

raw_dataset = load_dataset(path_to_raw)
raw_dataset = np.array(raw_dataset)

raw_dataset_shape = np.array(raw_dataset.shape)

generate_classification_file(raw_dataset=raw_dataset,
                             center_coordinates=coordinates,
                             label_name=label_name,
                             h5_output_path=output_h5_file_path,
                             subtomo_shape=subtomo_shape)

print("The script has finished!")
