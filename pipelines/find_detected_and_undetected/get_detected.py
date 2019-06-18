from os.path import join
from os import makedirs
import numpy as np

from src.python.calculator.statistics import \
    precision_recall_calculator_and_detected
from src.python.coordinates_toolbox.utils import \
    extract_coordinates_from_em_motl
from src.python.filereaders.csv import read_motl_from_csv
from src.python.filereaders.em import read_em
from src.python.peak_toolbox.utils import \
    extract_motl_coordinates_and_score_values
from src.python.filewriters.csv import unique_coordinates_motl_writer

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
parser.add_argument("-min_peak_distance", "--min_peak_distance",
                    type=int)
parser.add_argument("-label", "--label_name",
                    type=str)
parser.add_argument("-x_shift", "--x_shift",
                    type=int)

args = parser.parse_args()
path_to_csv_motl = args.path_to_csv_motl
path_to_motl_clean = args.path_to_motl_clean
output_dir = args.output_dir
min_peak_distance = args.min_peak_distance
label_name = args.label_name
x_shift = args.x_shift

figures_dir = join(output_dir, "figures")
makedirs(name=figures_dir, exist_ok=True)

_, motl_true = read_em(path_to_emfile=path_to_motl_clean)
motl_clean_coords = extract_coordinates_from_em_motl(motl_true)

motl_predicted = read_motl_from_csv(path_to_csv_motl)
motl_values, motl_coordinates = extract_motl_coordinates_and_score_values(
    motl_predicted)
del motl_predicted
#Todo change for 005
motl_coordinates = [point + np.array([x_shift, 0, 0]) for point in
                    motl_coordinates]

precision, recall, detected_true, detected_predicted, undetected_predicted, \
value_detected_predicted, value_undetected_predicted = \
    precision_recall_calculator_and_detected(
        motl_coordinates,
        motl_values,
        motl_clean_coords,
        radius=min_peak_distance)

print("len(detected_predicted)", len(detected_predicted))
print("len(undetected_predicted)", len(undetected_predicted))
#Todo change for 005
detected_predicted = [np.array(point) + np.array([-x_shift, 0, 0]) for point in
                      detected_predicted]
undetected_predicted = [np.array(point) + np.array([-x_shift, 0, 0]) for point
                        in undetected_predicted]

# detected_predicted = [np.array(point) for point in detected_predicted]
# undetected_predicted = [np.array(point) for point in undetected_predicted]



detected_output_folder = join(output_dir, "detected")
makedirs(name=detected_output_folder, exist_ok=True)
unique_coordinates_motl_writer(path_to_output_folder=detected_output_folder,
                               list_of_peak_scores=value_detected_predicted,
                               list_of_peak_coords=detected_predicted,
                               number_peaks_to_uniquify=-1,
                               minimum_peaks_distance=min_peak_distance)

undetected_output_folder = join(output_dir, "undetected")
makedirs(name=undetected_output_folder, exist_ok=True)
unique_coordinates_motl_writer(path_to_output_folder=undetected_output_folder,
                               list_of_peak_scores=value_undetected_predicted,
                               list_of_peak_coords=undetected_predicted,
                               number_peaks_to_uniquify=-1,
                               minimum_peaks_distance=min_peak_distance)
