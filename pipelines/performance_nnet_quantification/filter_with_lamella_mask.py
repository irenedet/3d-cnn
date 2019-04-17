import numpy as np
from os import makedirs
from src.python.filereaders.hdf import _load_hdf_dataset
from src.python.filereaders.csv import read_motl_from_csv
from src.python.filewriters.csv import motl_writer

lamella_file = "/scratch/trueba/3d-cnn/clean/180426_004/004_lamellamask_subtomo.hdf"
csv_motl = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_004/multi-class/G_s1_D4_IF8_w_1_64_1200_250/motl_3266_class_2.csv"
conserved_points_dir = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_004/multi-class/G_s1_D4_IF8_w_1_64_1200_250/fas/in_lamella_wide"
discarded_points_dir = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_004/multi-class/G_s1_D4_IF8_w_1_64_1200_250/fas/outside_lamella_wide"
border_thickness = 20
lamella_border = 40 # because lamella mask does not cover all the region of interest
x_shape = 928
y_shape = 928
z_shape = 221
z_shift = 380
makedirs(name=conserved_points_dir, exist_ok=True)
makedirs(name=discarded_points_dir, exist_ok=True)

lamella_indicator = _load_hdf_dataset(hdf_file_path=lamella_file)
motl_predicted = read_motl_from_csv(path_to_csv_motl=csv_motl)

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
    z_up = z - z_shift + lamella_border
    z_down = z - z_shift - lamella_border
    lamella_border_up = np.min([z_up, z_shape - 1])
    lamella_border_down = np.max([z_down, 0])
    if lamella_indicator[z - z_shift, y, x] == 1 and np.min(
            [x, y, x_shape - x, y_shape - y]) > border_thickness:
        conserved_values += [value]
        conserved_points += [point]
    elif lamella_indicator[lamella_border_up, y, x] == 1 and np.min(
            [x, y, x_shape - x, y_shape - y]) > border_thickness:
        conserved_values += [value]
        conserved_points += [point]
    elif lamella_indicator[lamella_border_down, y, x] == 1 and np.min(
            [x, y, x_shape - x, y_shape - y]) > border_thickness:
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
