from os import makedirs
import numpy as np

import matplotlib.pyplot as plt
from src.python.filereaders.csv import read_motl_from_csv
from src.python.peak_toolbox.utils import \
    extract_motl_coordinates_and_score_values
from src.python.filewriters.csv import unique_coordinates_motl_writer

# output_dir = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_006/mixed_trainset/no_shuffle/G1_confs_4_5_/peaks_in_training_partition/undetected/chosen/"
<<<<<<< HEAD
output_dir = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_004/multi-class/G_sigma1_D4_IF8/fas/in_lamella_wide/undetected/chosen"
makedirs(name=output_dir, exist_ok=True)

# path_to_csv_motl = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_006/mixed_trainset/no_shuffle/G1_confs_4_5_/peaks_in_training_partition/undetected/motl_3764.csv"
path_to_csv_motl = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_004/multi-class/G_sigma1_D4_IF8/fas/in_lamella_wide/undetected/motl_2997_class_0.csv"
=======
output_dir = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_005/confs_4_5_bis_/peaks_in_training_partition/undetected/chosen"
makedirs(name=output_dir, exist_ok=True)

# path_to_csv_motl = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_006/mixed_trainset/no_shuffle/G1_confs_4_5_/peaks_in_training_partition/undetected/motl_3764.csv"
path_to_csv_motl = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_005/confs_4_5_bis_/peaks_in_training_partition/undetected/motl_2511.csv"
>>>>>>> a989e851b8c49f42fa934d68991d6e56448b2c0f
motl_predicted = read_motl_from_csv(path_to_csv_motl)
motl_values, motl_coordinates = extract_motl_coordinates_and_score_values(
    motl_predicted)
del motl_predicted

# print(len(np.where(np.array(motl_values) > 1.33)[0]))
# print(np.min(motl_values[:500]))
# plt.hist(motl_values)
# plt.show()
<<<<<<< HEAD
threshold = 7
=======
threshold = 1
>>>>>>> a989e851b8c49f42fa934d68991d6e56448b2c0f
min_peak_distance = 12
# thresholded_indices = np.where(np.array(motl_values) > threshold)[0]
numb = len(np.where(np.array(motl_values) > threshold)[0])
motl_values = motl_values[:numb]
motl_coordinates = motl_coordinates[:numb]

z_coord = [point[0] for point in motl_coordinates]
print(np.max(z_coord))
print(np.min(z_coord))
chosen_coordinates = []
chosen_values = []
for val, point in zip(motl_values, motl_coordinates):
    z, y, x = point
<<<<<<< HEAD
    chosen_coordinates.append(point)
    chosen_values.append(val)
=======
    if (y < 400 or z > 370 + 70) and (y > 500 or z < 370 + 200):
        chosen_coordinates.append(point)
        chosen_values.append(val)
>>>>>>> a989e851b8c49f42fa934d68991d6e56448b2c0f

plt.hist(chosen_values, bins=12)
plt.show()
print("len(chosen_coordinates) = ", len(chosen_coordinates))
unique_coordinates_motl_writer(path_to_output_folder=output_dir,
                               list_of_peak_scores=chosen_values,
                               list_of_peak_coords=chosen_coordinates,
                               number_peaks_to_uniquify=-1,
                               minimum_peaks_distance=min_peak_distance)

plt.hist(motl_values)
plt.show()
