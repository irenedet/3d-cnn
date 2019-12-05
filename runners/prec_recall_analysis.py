import argparse
import os
from os import makedirs
from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from calculator.statistics import precision_recall_calculator_and_detected, \
    pr_auc_score, F1_score_calculator
from coordinates_toolbox.utils import extract_coordinates_from_em_motl
from filereaders.csv import read_motl_from_csv
from filereaders.em import read_em
from filewriters.csv import motl_writer
from peak_toolbox.utils import extract_motl_coordinates_and_score_values

parser = argparse.ArgumentParser()

parser.add_argument("-output", "--output_dir",
                    help="directory where the outputs will be stored",
                    type=str)
parser.add_argument("-motl", "--path_to_motl",
                    help="path to the motive list in csv format",
                    type=str)
parser.add_argument("-clean", "--path_to_clean",
                    help="path to the motive list of true particles",
                    type=str)
parser.add_argument("-label", "--label_name",
                    help="name of segmentation",
                    type=str)
parser.add_argument("-min_peak_distance", "--min_peak_distance",
                    help="radius in pixels to be considered same particle",
                    type=int)
parser.add_argument("-x_shift", "--x_shift",
                    help="shift between motls in the x coordinate",
                    type=int)

args = parser.parse_args()
output_dir = args.output_dir
path_to_csv_motl = args.path_to_motl
path_to_motl_clean = args.path_to_clean
label_name = args.label_name
min_peak_distance = args.min_peak_distance
x_shift = args.x_shift

figures_dir = join(output_dir, "figures")
makedirs(name=figures_dir, exist_ok=True)
# Extract coordinates from template matching

_, motl_extension = os.path.splitext(path_to_motl_clean)
assert motl_extension == ".em" or motl_extension == ".csv"
if motl_extension == ".em":
    print("motl clean in .em format")
    header, motl_true = read_em(path_to_emfile=path_to_motl_clean)
    motl_clean_coords = extract_coordinates_from_em_motl(motl_true)
else:
    print("motl clean in .csv format")
    motl_true = read_motl_from_csv(path_to_motl_clean)
    _, motl_clean_coords = extract_motl_coordinates_and_score_values(
        motl_true)
    motl_clean_coords = np.array(motl_clean_coords)

motl_predicted = read_motl_from_csv(path_to_csv_motl)
motl_values, motl_coordinates = extract_motl_coordinates_and_score_values(
    motl_predicted)

# motl_values /= np.max(motl_values)
n = 5000
motl_values, motl_coordinates = motl_values[:n], motl_coordinates[:n]

print("motl_clean_coords.shape", motl_clean_coords.shape)
motl_coordinates = [[row[0] + x_shift, row[1], row[2]] for row in
                    motl_coordinates]

precision, recall, detected_true, detected_predicted, \
undetected_predicted, value_detected_predicted, \
value_undetected_predicted, redundantly_detected_predicted, \
value_redudndantly_detected_predicted = \
    precision_recall_calculator_and_detected(
        motl_coordinates,
        motl_values,
        motl_clean_coords,
        radius=min_peak_distance)

detected_predicted = [np.array(point) for point in detected_predicted]

path_to_detected_predicted = join(output_dir, "detected")
path_to_undetected_predicted = join(output_dir, "undetected")
path_to_redundantly_detected = join(output_dir, "redundant")
makedirs(name=path_to_detected_predicted, exist_ok=True)
makedirs(name=path_to_undetected_predicted, exist_ok=True)
makedirs(name=path_to_redundantly_detected, exist_ok=True)

from filewriters.csv import filter_duplicate_values_by_score

# ToDo: add as a parameter for filtering:
max_peaks_distance = 8

value_detected_predicted, detected_predicted = filter_duplicate_values_by_score(
    list_of_peak_coords=detected_predicted,
    list_of_peak_scores=value_detected_predicted,
    number_peaks_to_uniquify=len(detected_predicted),
    minimum_peaks_distance=max_peaks_distance,
    uniquify_by_score=True)
value_undetected_predicted, undetected_predicted = filter_duplicate_values_by_score(
    list_of_peak_coords=undetected_predicted,
    list_of_peak_scores=value_undetected_predicted,
    number_peaks_to_uniquify=len(undetected_predicted),
    minimum_peaks_distance=max_peaks_distance,
    uniquify_by_score=True)

motl_writer(path_to_output_folder=path_to_redundantly_detected,
            list_of_peak_coords=redundantly_detected_predicted,
            list_of_peak_scores=value_redudndantly_detected_predicted,
            in_tom_format=True)
motl_writer(path_to_output_folder=path_to_detected_predicted,
            list_of_peak_coords=detected_predicted,
            list_of_peak_scores=value_detected_predicted,
            in_tom_format=True)
motl_writer(path_to_output_folder=path_to_undetected_predicted,
            list_of_peak_coords=undetected_predicted,
            list_of_peak_scores=value_undetected_predicted,
            in_tom_format=True)

matplotlib.use('Agg')
plt.ioff()
plt.figure(1)
plt.hist(motl_values, bins=30, label="Predicted particles")
plt.xlabel("score value")
plt.ylabel("frequency")
plt.title(str(len(motl_coordinates)) + " peaks, " + label_name)
plt.legend()
plt.gcf()
figure_name = join(figures_dir, "histogram.png")
plt.savefig(fname=figure_name,
            format="png")

matplotlib.use('Agg')
plt.ioff()
plt.figure(2)
plt.hist(value_detected_predicted, bins=25, label="true positives",
         fc=(0, 0, 1, 0.5))
plt.hist(value_undetected_predicted, bins=25, label="false positives",
         fc=(1, 0, 0, 0.5))
plt.xlabel("score value")
plt.ylabel("frequency")
plt.title(str(len(motl_coordinates)) + " peaks, " + label_name)
plt.legend()
plt.gcf()
figure_name = join(figures_dir, "histogram-detected-undetected.png")
plt.savefig(fname=figure_name,
            format="png")

F1_score = F1_score_calculator(precision, recall)
max_F1 = np.max(F1_score)
optimal_peak_number = np.min(np.where(F1_score == max_F1)[0])
auPRC = pr_auc_score(precision=precision, recall=recall)
print("auPRC = ", auPRC, "max_F1 = ", max_F1)

pr_legend_str = label_name + " , auPRC = " + str(round(auPRC, 4))
f1_legend_str = label_name + " , (max_F1, best_peaks) = (" + str(
    round(max_F1, 4)) + ", " + str(
    optimal_peak_number) + ")"

plt.figure(3)
plt.plot(F1_score, label=f1_legend_str)
plt.xlabel("number of peaks")
plt.ylabel("F1 score")
plt.title(str(len(motl_coordinates)) + " peaks, " + label_name)
plt.legend()
plt.gcf()
figure_name = join(figures_dir, "F1_score.png")
plt.savefig(fname=figure_name,
            format="png")

plt.figure(4)
plt.plot(recall, precision, label=pr_legend_str)
plt.xlabel("recall")
plt.ylabel("precision")
plt.title(str(len(motl_coordinates)) + " peaks, " + label_name)
plt.legend()
plt.gcf()
figure_name = join(figures_dir, "PR_curve.png")
plt.savefig(fname=figure_name,
            format="png")

print("Figures saved at ", figures_dir)
