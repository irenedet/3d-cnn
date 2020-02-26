import matplotlib.pyplot as plt
import numpy as np
import time
import os
from os import makedirs
from os.path import join

from src.python.peak_toolbox.subtomos import select_coordinates_in_subtomos, \
    select_coordinates_and_values_in_subtomos
from src.python.calculator.statistics import pr_auc_score, \
    f1_score_calculator, precision_recall_calculator_and_detected
from src.python.coordinates_toolbox.utils import \
    extract_coordinates_from_em_motl
from src.python.filereaders.csv import read_motl_from_csv
from src.python.filereaders.em import read_em
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
parser.add_argument("-test_file", "--testing_set_data_path",
                    help="path to h5 file with test subtomos format",
                    type=str)
parser.add_argument("-output", "--output_dir",
                    help="directory where the outputs will be stored",
                    type=str)
parser.add_argument("-radius", "--sphere_radius",
                    type=int)
parser.add_argument("-z_shift", "--output_z_shift",
                    type=int)
parser.add_argument("-x_shift", "--output_x_shift",
                    type=int)
parser.add_argument("-shape_x", "--output_shape_x",
                    type=int)
parser.add_argument("-shape_y", "--output_shape_y",
                    type=int)
parser.add_argument("-shape_z", "--output_shape_z",
                    type=int)
parser.add_argument("-box", "--subtomo_box_length",
                    type=int)
parser.add_argument("-overlap", "--overlap",
                    default=12,
                    type=int)

args = parser.parse_args()
path_to_csv_motl = args.path_to_csv_motl
path_to_motl_clean = args.path_to_motl_clean
output_dir = args.output_dir
radius = args.sphere_radius
testing_set_data_path = args.testing_set_data_path
shape_x = args.output_shape_x
shape_y = args.output_shape_y
shape_z = args.output_shape_z
z_shift = args.output_z_shift
x_shift = args.output_x_shift
box = args.subtomo_box_length
overlap = args.overlap

print("")
print("output_dir = ", output_dir)
# print("testing_set_data_path = ", testing_set_data_path)
# print("path_to_csv_motl = ", path_to_csv_motl)

dataset_shape = (shape_z, shape_y, shape_x)
subtomo_shape = (box - overlap, box - overlap, box - overlap)

figures_dir = join(output_dir, "figures")
makedirs(name=figures_dir, exist_ok=True)

motl_predicted = read_motl_from_csv(path_to_csv_motl)


# ToDo checking this
# Header, motl_true = read_em(path_to_emfile=path_to_motl_clean)
# true_coordinates = extract_coordinates_from_em_motl(motl_true)

# print("len(true_coordinates) = ", len(true_coordinates))
# print("min_z", np.min([point[2] for point in true_coordinates]))
_, motl_extension = os.path.splitext(path_to_motl_clean)

if motl_extension == ".em":
    print("motl clean in .em format")
    Header, motl_true = read_em(path_to_emfile=path_to_motl_clean)
    true_coordinates = extract_coordinates_from_em_motl(motl_true)
elif motl_extension == ".csv":
    print("motl clean in .csv format")
    motl_true = read_motl_from_csv(path_to_motl_clean)
    _, true_coordinates = extract_motl_coordinates_and_score_values(
        motl_true)
    true_coordinates = np.array(true_coordinates)
    del motl_true
else:
    print("motl clean should be in a valid format .em or .csv")


predicted_coordinates = [np.array([row[7], row[8], row[9]]) for row in
                         motl_predicted]
motl_values = [row[0] for row in motl_predicted]
del motl_predicted
unique_peaks_number = len(motl_values)

start = time.time()
true_coordinates_test, _ = \
    select_coordinates_in_subtomos(
        coordinates=true_coordinates,
        subtomo_file_path=testing_set_data_path,
        split=0,
        data_order="same",
        dataset_shape=dataset_shape,
        subtomo_shape=subtomo_shape,
        shift=z_shift)
true_coordinates_test = np.array(true_coordinates_test)
end = time.time()
# print("elapsed time for splitting test and train true coordinates", end - start,
#       "sec")
# print("len(true_coordinates_test) = ", len(true_coordinates_test))

start = time.time()
predicted_coordinates_test, _, test_predicted_coordinates_values, _ = \
    select_coordinates_and_values_in_subtomos(
        coordinates=predicted_coordinates,
        values=motl_values,
        subtomo_file_path=testing_set_data_path,
        split=0,
        data_order="same",
        dataset_shape=dataset_shape,
        subtomo_shape=subtomo_shape,
        shift=z_shift)
end = time.time()
# print("elapsed time for splitting test and train predicted coordinates",
#       end - start, "sec")
# print("len(predicted_coordinates_test) = ", len(predicted_coordinates_test))

y_shift = 0
z_shift = 0
shift_vector = np.array([x_shift, y_shift, z_shift])
predicted_coordinates_test = [np.array(p) + shift_vector for p in
                              predicted_coordinates_test]
predicted_coordinates_test = np.array(predicted_coordinates_test)

precision, recall, detected_clean, detected_predicted, \
undetected_predicted, value_detected_predicted, value_undetected_predicted = \
    precision_recall_calculator_and_detected(
        predicted_coordinates_test,
        test_predicted_coordinates_values,
        true_coordinates_test,
        radius=radius)

# print("len(test_predicted_coordinates_values)",
#       len(test_predicted_coordinates_values))

thresholded_predicted_indices = \
    np.where(np.array(test_predicted_coordinates_values) > 0)[0]
thresholded_predicted_values = [
    test_predicted_coordinates_values[index] for index in
    thresholded_predicted_indices]
plt.figure(1)
plt.hist(thresholded_predicted_values, bins=10, label="thresholded predicted")
plt.xlabel("score value")
plt.ylabel("frequency")
plt.title(str(len(thresholded_predicted_indices)) +
          " peaks")
plt.legend()
plt.gcf()
fig_name = join(figures_dir, "histogram_testset.png")
plt.savefig(fname=fig_name,
            format="png")

# to do, in dice multi class should be softmax!
sigmoid = lambda t: 1 / (1 + np.exp(-t))

sigmoid_predicted_coordinates_test_values = [sigmoid(t) for t in
                                             thresholded_predicted_values]
plt.figure(2)
plt.hist(sigmoid_predicted_coordinates_test_values, bins=10,
         label="thresholded predicted")
plt.xlabel("sigmoid(score value)")
plt.ylabel("frequency")
plt.title(str(len(predicted_coordinates))
          + " peaks")
plt.legend()
plt.gcf()
fig_name = join(figures_dir, "histogram_sigmoid_testset.png")
plt.savefig(fname=fig_name,
            format="png")

thresholded_predicted_indices = \
    np.where(np.array(value_detected_predicted) > 0)[0]
value_detected_predicted_range = [value_detected_predicted[index] for index in
                                  thresholded_predicted_indices]
thresholded_predicted_indices = \
    np.where(np.array(value_undetected_predicted) > 0)[0]
value_undetected_predicted_range = [value_undetected_predicted[index] for index
                                    in thresholded_predicted_indices]
plt.figure(33)
plt.hist(value_detected_predicted_range, bins=15, label="true positives",
         fc=(0, 0, 1, 0.5))
plt.hist(value_undetected_predicted_range,
         bins=15, label="false positives",
         fc=(1, 0, 0, 0.5))
plt.xlabel("score value")
plt.ylabel("frequency")
plt.title(str(len(thresholded_predicted_indices))
          + " peaks")
plt.legend()
plt.gcf()
fig_name = join(figures_dir,
                "histogram-detected-undetected_testset.png")
plt.savefig(fname=fig_name,
            format="png")

thresholded_predicted_indices = \
    np.where(np.array(value_detected_predicted) > 0.1)[0]
value_detected_predicted_range = [value_detected_predicted[index] for index in
                                  thresholded_predicted_indices]
thresholded_predicted_indices = \
    np.where(np.array(value_undetected_predicted) > 0.1)[0]
value_undetected_predicted_range = [value_undetected_predicted[index] for index
                                    in thresholded_predicted_indices]
plt.figure(3)
plt.hist(value_detected_predicted_range, bins=15, label="true positives",
         fc=(0, 0, 1, 0.5))
plt.hist(value_undetected_predicted_range,
         bins=15, label="false positives",
         fc=(1, 0, 0, 0.5))
plt.xlabel("score value")
plt.ylabel("frequency")
plt.title(str(len(thresholded_predicted_indices))
          + " peaks, threshold on score = 0.1")
plt.legend()
plt.gcf()
fig_name = join(figures_dir,
                "histogram-detected-undetected_thresholded_testset.png")
plt.savefig(fname=fig_name,
            format="png")

sigmoid_detected_predicted_values = [sigmoid(value) for value in
                                     value_detected_predicted_range]
sigmoid_undetected_predicted_values = [sigmoid(value) for value in
                                       value_undetected_predicted_range]
plt.figure(4)
plt.hist(sigmoid_detected_predicted_values, bins=15, label="true positives",
         fc=(0, 0, 1, 0.5))
plt.hist(sigmoid_undetected_predicted_values,
         bins=15, label="false positives",
         fc=(1, 0, 0, 0.5))
plt.xlabel("sigmoid(score value)")
plt.ylabel("frequency")
plt.title(str(len(predicted_coordinates)) +
          " peaks, threshold on score = 0.02")
plt.legend()
plt.gcf()
fig_name = join(figures_dir,
                "histogram-sigmoid-detected-undetected_thr_testset.png")
plt.savefig(fname=fig_name,
            format="png")

F1_score = f1_score_calculator(precision, recall)
max_F1 = np.max(F1_score)
optimal_peak_number = np.min(np.where(F1_score == max_F1)[0])
auPRC = pr_auc_score(precision=precision, recall=recall)

print("auPRC = ", auPRC, "and max_F1 = ", max_F1)

pr_legend_str = "auPRC = " + str(round(auPRC, 4))
f1_legend_str = "(max_F1, best_peaks) = (" + str(
    round(max_F1, 4)) + ", " + str(
    optimal_peak_number) + ")"
title_str = str(len(predicted_coordinates_test)) + " peaks"
plt.figure(5)
plt.plot(F1_score, label=f1_legend_str)
plt.xlabel("number of peaks")
plt.ylabel("F1 score")
plt.title(title_str)
plt.legend()
plt.gcf()
fig_name = join(figures_dir, "f1_score_" + title_str + ".png")
plt.savefig(fname=fig_name,
            format="png")
plt.figure(6)
plt.plot(recall, precision, label=pr_legend_str)
plt.xlabel("recall")
plt.ylabel("precision")
plt.title(title_str)
plt.legend()
plt.gcf()
fig_name = join(figures_dir, "pr_" + title_str + ".png")
plt.savefig(fname=fig_name,
            format="png")

print("All plots saved in ", figures_dir)
