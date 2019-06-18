import numpy as np
import h5py

from src.python.calculator.statistics import \
    precision_recall_calculator_and_detected, \
    pr_auc_score, \
    F1_score_calculator
from src.python.coordinates_toolbox.utils import \
    extract_coordinates_from_em_motl
from src.python.filereaders.csv import read_motl_from_csv
from src.python.filereaders.em import read_em
from src.python.peak_toolbox.utils import \
    extract_motl_coordinates_and_score_values

path_to_motl = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_005/confs_8_5_bis_/motl_4527.csv"

from src.python.filereaders.csv import read_motl_from_csv

motl = read_motl_from_csv(path_to_motl)
coordinates = [np.array([row[7], row[8], row[9] + 370]) for row in motl[:100]]

for i in range(3):
    max = np.max([point[i] for point in coordinates])
    print(i, max)

path_to_motl_clean = '/scratch/trueba/3d-cnn/clean/180426_005/motl_clean_4b.em'

# Extract coordinates from template matching
Header, motl_clean = read_em(path_to_emfile=path_to_motl_clean)
motl_clean_coords = extract_coordinates_from_em_motl(motl_clean)
motl_clean_coords[:, 0] += np.ones(motl_clean_coords.shape[0])
print(motl_clean_coords.shape)

for i in range(3):
    max = np.max([point[i] for point in motl_clean_coords])
    print(i, max)
