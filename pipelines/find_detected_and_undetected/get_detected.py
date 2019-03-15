from os.path import join
from os import makedirs

from src.python.calculator.statistics import \
    precision_recall_calculator_and_detected
from src.python.coordinates_toolbox.utils import \
    extract_coordinates_from_em_motl
from src.python.filereaders.csv import read_motl_from_csv
from src.python.filereaders.em import load_em_motl
from src.python.peak_toolbox.utils import \
    extract_motl_coordinates_and_score_values

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-motl", "--path_to_csv_motl",
                    help="path to motive list in .em or .csv format",
                    type=str)
parser.add_argument("-clean", "--path_to_motl_clean",
                    help="path to motive list in .em or .csv format",
                    type=str)
parser.add_argument("-output", "--output_dir",
                    help="directory where the outputs will be stored",
                    type=str)
parser.add_argument("-radius", "--sphere_radius",
                    type=int)
parser.add_argument("-label", "--label_name",
                    type=str)

args = parser.parse_args()
path_to_csv_motl = args.path_to_csv_motl
path_to_motl_clean = args.path_to_motl_clean
output_dir = args.output_dir
radius = args.sphere_radius
label_name = args.label_name

# label_name = "ribosomes"
# output_dir = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_006/confs_4_5_bis_/"
# path_to_csv_motl = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_006/confs_4_5_bis_/motl_4896.csv"
# path_to_motl_clean = '/scratch/trueba/3d-cnn/clean/180426_006/motl_clean_4b.em'

figures_dir = join(output_dir, "figures")
makedirs(name=figures_dir, exist_ok=True)

_, motl_true = load_em_motl(path_to_emfile=path_to_motl_clean)
motl_clean_coords = extract_coordinates_from_em_motl(motl_true)

motl_predicted = read_motl_from_csv(path_to_csv_motl)
motl_values, motl_coordinates = extract_motl_coordinates_and_score_values(
    motl_predicted)
del motl_predicted

precision, recall, detected_true, detected_predicted, undetected_predicted, \
value_detected_predicted, value_undetected_predicted = \
    precision_recall_calculator_and_detected(
        motl_coordinates,
        motl_values,
        motl_clean_coords,
        radius=radius)

# detected_predicted = [np.array(point) for point in detected_predicted]
# undetected_predicted = [np.array(point) for point in undetected_predicted]

from src.python.filewriters.csv import unique_coordinates_motl_writer

detected_output_folder = join(output_dir, "detected")
makedirs(name=detected_output_folder, exist_ok=True)
unique_coordinates_motl_writer(path_to_output_folder=detected_output_folder,
                               list_of_peak_scores=value_detected_predicted,
                               list_of_peak_coords=detected_predicted,
                               number_peaks_to_uniquify=5000,
                               minimum_peaks_distance=radius)

undetected_output_folder = join(output_dir, "undetected")
makedirs(name=undetected_output_folder, exist_ok=True)
unique_coordinates_motl_writer(path_to_output_folder=undetected_output_folder,
                               list_of_peak_scores=value_undetected_predicted,
                               list_of_peak_coords=undetected_predicted,
                               number_peaks_to_uniquify=5000,
                               minimum_peaks_distance=radius)
