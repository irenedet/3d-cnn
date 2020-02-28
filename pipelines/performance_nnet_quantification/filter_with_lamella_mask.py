import numpy as np
from os import makedirs
from os.path import join
from file_actions.readers.hdf import _load_hdf_dataset
from file_actions.readers.csv import read_motl_from_csv
from file_actions.writers.csv import motl_writer

# lamella_file = "/scratch/trueba/3d-cnn/clean/180426_004/004_lamellamask_subtomo.hdf"
# csv_motl = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_004/multi-class/G_s1_D4_IF8_w_1_64_1200_250/motl_3266_class_2.csv"
# output_dir = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_004/multi-class/G_s1_D4_IF8_w_1_64_1200_250/fas"
# dataset_border_xy = 20
# lamella_extension = 40 # because lamella mask does not cover all the region of interest
# x_dim = 928
# y_dim = 928
# z_dim = 221
# z_shift = 380

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-output_dir", "--output_dir",
                    help="path to the output directory",
                    type=str)
parser.add_argument("-lamella_file", "--lamella_file",
                    help="path to lamella mask file",
                    type=str)
parser.add_argument("-csv_motl", "--csv_motl",
                    help="path to motl in csv format",
                    type=str)
parser.add_argument("-border_xy", "--dataset_border_xy",
                    help="border thickness in xy plane to discard particles",
                    type=int)
parser.add_argument("-lamella_extension", "--lamella_extension",
                    help="pixels up and down the lamella that will be added",
                    type=int)
parser.add_argument("-x_dim", "--output_xdim",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-y_dim", "--output_ydim",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-z_dim", "--output_zdim",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-z_shift", "--z_shift_original",
                    help="name of category to be segmented",
                    type=int)

args = parser.parse_args()
lamella_file = args.lamella_file
csv_motl = args.csv_motl
# x_dim = args.output_xdim
# y_dim = args.output_ydim
# z_dim = args.output_zdim
dataset_border_xy = args.dataset_border_xy
lamella_extension = args.lamella_extension
output_dir = args.output_dir
z_shift = args.z_shift_original

conserved_points_dir = join(output_dir, "in_lamella")
discarded_points_dir = join(output_dir, "outside_lamella")
makedirs(name=conserved_points_dir, exist_ok=True)
makedirs(name=discarded_points_dir, exist_ok=True)

lamella_indicator = _load_hdf_dataset(hdf_file_path=lamella_file)
motl_predicted = read_motl_from_csv(path_to_csv_motl=csv_motl)

lamella_indicator = np.array(lamella_indicator)
z_dim, y_dim, x_dim = lamella_indicator.shape
print("(x_dim, y_dim, z_dim) =", x_dim, y_dim, z_dim)

motl_values = [row[0] for row in motl_predicted]
predicted_coordinates = [np.array([row[7], row[8], row[9]]) for row in
                         motl_predicted]

conserved_points = []
conserved_values = []
discarded_points = []
discarded_values = []
for value, point in zip(motl_values, predicted_coordinates):
    point = [int(entry) for entry in point]
    x, y, z = point
    z_up = z - z_shift + lamella_extension
    z_down = z - z_shift - lamella_extension
    lamella_border_up = np.min([z_up, z_dim - 1])
    lamella_border_down = np.max([z_down, 0])
    if lamella_indicator[z - z_shift, y, x] == 1 and np.min(
            [x, y, x_dim - x, y_dim - y]) > dataset_border_xy:
        conserved_values += [value]
        conserved_points += [point]
    elif lamella_indicator[lamella_border_up, y, x] == 1 and np.min(
            [x, y, x_dim - x, y_dim - y]) > dataset_border_xy:
        conserved_values += [value]
        conserved_points += [point]
    elif lamella_indicator[lamella_border_down, y, x] == 1 and np.min(
            [x, y, x_dim - x, y_dim - y]) > dataset_border_xy:
        conserved_values += [value]
        conserved_points += [point]
    else:
        discarded_points += [point]
        discarded_values += [value]

motl_writer(path_to_output_folder=conserved_points_dir,
            list_of_peak_scores=conserved_values,
            list_of_peak_coords=conserved_points,
            in_tom_format=True)
motl_writer(path_to_output_folder=discarded_points_dir,
            list_of_peak_scores=discarded_values,
            list_of_peak_coords=discarded_points,
            in_tom_format=True)
