import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from os import makedirs
from os.path import join

from src.python.peak_toolbox.subtomos import select_coordinates_in_subtomos, \
    select_coordinates_and_values_in_subtomos
from src.python.calculator.statistics import pr_auc_score, \
    F1_score_calculator, precision_recall_calculator_and_detected
from src.python.peak_toolbox.utils import read_motl_coordinates_and_values
from src.python.filewriters.csv import motl_writer
# unique_coordinates_motl_writer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-dataset_table", "--dataset_table",
                    help="path to dataset_table",
                    type=str)
parser.add_argument("-class_name", "--class_name",
                    help="name of class, i.e. ribo or fas",
                    type=str)
parser.add_argument("-statistics_file", "--statistics_file",
                    help="file where auPRC will be recorded.",
                    type=str)
parser.add_argument("-label_name", "--label_name",
                    help="name of class, i.e. ribo or fas",
                    type=str)
parser.add_argument("-tomo_name", "--tomo_name",
                    help="tomo to be analyzed",
                    type=str)
parser.add_argument("-motl", "--path_to_motl_predicted",
                    help="path to motive list in .em or .csv format",
                    type=str)
parser.add_argument("-clean", "--path_to_motl_true",
                    help="path to motive list in .em or .csv format",
                    type=str)
parser.add_argument("-test_file", "--testing_set_data_path",
                    help="path to h5 file with test subtomos format",
                    type=str,
                    default="None")
parser.add_argument("-output", "--output_dir",
                    help="directory where the outputs will be stored",
                    type=str)
parser.add_argument("-radius", "--sphere_radius",
                    type=int)
parser.add_argument("-z_shift", "--output_z_shift",
                    type=int,
                    default=0)
parser.add_argument("-y_shift", "--output_y_shift",
                    type=int,
                    default=0)
parser.add_argument("-x_shift", "--output_x_shift",
                    type=int,
                    default=0)
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
parser.add_argument("-threshold", "--score_threshold",
                    default=-10,
                    type=float)
parser.add_argument("-summary_file", "--summary_file",
                    type=str)

args = parser.parse_args()
dataset_table = args.dataset_table
tomo_name = args.tomo_name
path_to_motl_predicted = args.path_to_motl_predicted
output_dir = args.output_dir
radius = args.sphere_radius
box = args.subtomo_box_length
overlap = args.overlap
threshold = args.score_threshold
tight_threshold = 0.1
class_name = args.class_name
statistics_file = args.statistics_file
label_name = args.label_name
summary_file = args.summary_file

df = pd.read_csv(dataset_table)
df['tomo_name'] = df['tomo_name'].astype(str)
tomo_df = df[df['tomo_name'] == tomo_name]
z_shift = int(tomo_df.iloc[0]['z_shift'])
y_shift = int(tomo_df.iloc[0]['y_shift'])
x_shift = int(tomo_df.iloc[0]['x_shift'])
x_dim = int(tomo_df.iloc[0]['x_dim'])
y_dim = int(tomo_df.iloc[0]['y_dim'])
z_dim = int(tomo_df.iloc[0]['z_dim'])
test_partition = tomo_df.iloc[0]['test_partition']
clean_motive_list_name = 'path_to_motl_clean_' + class_name
path_to_motl_true = tomo_df.iloc[0][clean_motive_list_name]

print("")
print("path_to_motl_true = ", path_to_motl_true)
print("output_dir = ", output_dir)
dataset_shape = (z_dim, y_dim, x_dim)
subtomo_shape = (box - overlap, box - overlap, box - overlap)

figures_dir = join(output_dir, "figures")
makedirs(name=figures_dir, exist_ok=True)

predicted_values, predicted_coordinates = read_motl_coordinates_and_values(
    path_to_motl=path_to_motl_predicted)

true_values, true_coordinates = read_motl_coordinates_and_values(
    path_to_motl=path_to_motl_true)
unique_peaks_number = len(predicted_values)

if test_partition == "None":
    print("all particles will be considered for testing...")
    true_coordinates_test = true_coordinates
    predicted_coordinates_test = predicted_coordinates
    predicted_values_test = predicted_values
else:
    print("Only particles in the test patiition will be considered",
          "for testing...")
    start = time.time()
    true_coordinates_test, _ = \
        select_coordinates_in_subtomos(
            coordinates=true_coordinates,
            subtomo_file_path=test_partition,
            split=0,
            data_order="same",
            dataset_shape=dataset_shape,
            subtomo_shape=subtomo_shape,
            shift=z_shift)
    true_coordinates_test = np.array(true_coordinates_test)
    end = time.time()

    start = time.time()
    predicted_coordinates_test, _, predicted_values_test, _ = \
        select_coordinates_and_values_in_subtomos(
            coordinates=predicted_coordinates,
            values=predicted_values,
            subtomo_file_path=test_partition,
            split=0,
            data_order="same",
            dataset_shape=dataset_shape,
            subtomo_shape=subtomo_shape,
            shift=z_shift)
    end = time.time()
z_shift = 0
shift_vector = np.array([x_shift, y_shift, z_shift])
predicted_coordinates_test = [np.array(p) + shift_vector for p in
                              predicted_coordinates_test]
predicted_coordinates_test = np.array(predicted_coordinates_test)

precision, recall, detected_true, detected_predicted, \
undetected_predicted, value_detected_predicted, \
value_undetected_predicted, redundantly_detected_predicted, \
value_redudndantly_detected_predicted = \
    precision_recall_calculator_and_detected(
        predicted_coordinates_test,
        predicted_values_test,
        true_coordinates_test,
        radius=radius)

F1_score = F1_score_calculator(precision, recall)
max_F1 = np.max(F1_score)
optimal_peak_number = np.min(np.where(F1_score == max_F1)[0])
auPRC = pr_auc_score(precision=precision, recall=recall)
print("auPRC = ", auPRC, "and max_F1 = ", max_F1)

path_to_detected_predicted = join(output_dir, "detected")
path_to_undetected_predicted = join(output_dir, "undetected")
makedirs(name=path_to_detected_predicted, exist_ok=True)
makedirs(name=path_to_undetected_predicted, exist_ok=True)

motl_writer(path_to_output_folder=path_to_detected_predicted,
            list_of_peak_coords=detected_predicted,
            list_of_peak_scores=value_detected_predicted,
            in_tom_format=True)
motl_writer(path_to_output_folder=path_to_undetected_predicted,
            list_of_peak_coords=undetected_predicted,
            list_of_peak_scores=value_undetected_predicted,
            in_tom_format=True)

# unique_coordinates_motl_writer(path_to_output_folder=output_dir,
#                                list_of_peak_scores=value_detected_predicted,
#                                list_of_peak_coords=detected_predicted,
#                                number_peaks_to_uniquify=len(
#                                    value_detected_predicted),
#                                minimum_peaks_distance=2*radius,
#                                class_number=None,
#                                in_tom_format=True,
#                                motl_name="motl_detetected.csv",
#                                uniquify_by_score=True)

# unique_coordinates_motl_writer(path_to_output_folder=output_dir,
#                                list_of_peak_scores=value_undetected_predicted,
#                                list_of_peak_coords=undetected_predicted,
#                                number_peaks_to_uniquify=len(
#                                    value_undetected_predicted),
#                                minimum_peaks_distance=2*radius,
#                                class_number=None,
#                                in_tom_format=True,
#                                motl_name="motl_undetetected.csv",
#                                uniquify_by_score=True)

if threshold == -10:
    threshold = np.min(predicted_values_test)
else:
    print("peak threshold set to ", threshold)

thresholded_predicted_indices = \
    np.where(np.array(predicted_values_test) >= threshold)[0]
thresholded_predicted_values = [predicted_values_test[index] for
                                index in thresholded_predicted_indices]

thresholded_detected_predicted_indices = \
    np.where(np.array(value_detected_predicted) >= threshold)[0]
value_detected_predicted_range = [value_detected_predicted[index] for index in
                                  thresholded_detected_predicted_indices]

thresholded_undetected_predicted_indices = \
    np.where(np.array(value_undetected_predicted) >= threshold)[0]
value_undetected_predicted_range = [value_undetected_predicted[index] for index
                                    in thresholded_undetected_predicted_indices]

tight_thresholded_detected_predicted_indices = \
    np.where(np.array(value_detected_predicted) >= tight_threshold)[0]

value_detected_predicted_tight = [value_detected_predicted[index] for index in
                                  tight_thresholded_detected_predicted_indices]

tight_thresholded_undetected_predicted_indices = \
    np.where(np.array(value_undetected_predicted) >= tight_threshold)[0]

value_undetected_predicted_tight = [
    value_undetected_predicted[index] for index in
    tight_thresholded_undetected_predicted_indices]

# First plot
plt.figure(1)
plt.hist(thresholded_predicted_values, bins=10, label="thresholded predicted")
plt.xlabel("score value")
plt.ylabel("frequency")
plt.title(str(len(thresholded_predicted_indices)) + " peaks")
plt.legend()
plt.gcf()
fig_name = join(figures_dir, "histogram_test_set.png")
plt.savefig(fname=fig_name,
            format="png")

# to do, in dice multi class should be softmax!
# sigmoid = lambda t: 1 / (1 + np.exp(-t))
#
# sigmoid_predicted_coordinates_test_values = [sigmoid(t) for t in
#                                              thresholded_predicted_values]
#
# Second plot
# plt.figure(2)
# plt.hist(sigmoid_predicted_coordinates_test_values, bins=10,
#          label="thresholded predicted")
# plt.xlabel("sigmoid(score value)")
# plt.ylabel("frequency")
# plt.title(str(len(predicted_coordinates)) + " peaks")
# plt.legend()
# plt.gcf()
# fig_name = join(figures_dir, "histogram_sigmoid_testset.png")
# plt.savefig(fname=fig_name,
#             format="png")

plt.figure(3)
plt.hist(value_detected_predicted_range, bins=15, label="true positives",
         fc=(0, 0, 1, 0.5))
plt.hist(value_undetected_predicted_range, bins=15, label="false positives",
         fc=(1, 0, 0, 0.5))
plt.xlabel("score value")
plt.ylabel("frequency")
plt.title(str(len(thresholded_undetected_predicted_indices)) + " peaks")
plt.legend()
plt.gcf()
fig_name = join(figures_dir, "histogram-detected-undetected_testset.png")
plt.savefig(fname=fig_name, format="png")

plt.figure(4)
plt.hist(value_detected_predicted_tight, bins=15, label="true positives",
         fc=(0, 0, 1, 0.5))
plt.hist(value_undetected_predicted_tight, bins=15, label="false positives",
         fc=(1, 0, 0, 0.5))
plt.xlabel("score value")
plt.ylabel("frequency")
plt.title(str(len(tight_thresholded_detected_predicted_indices) +
              len(tight_thresholded_undetected_predicted_indices))
          + " peaks, threshold on score = " + str(tight_threshold))
plt.legend()
plt.gcf()
fig_name = join(figures_dir,
                "histogram-detected-undetected_thresholded_testset.png")
plt.savefig(fname=fig_name,
            format="png")

# sigmoid_detected_predicted_values = [sigmoid(value) for value in
#                                      value_detected_predicted_tight]
# sigmoid_undetected_predicted_values = [sigmoid(value) for value in
#                                        value_undetected_predicted_tight]
#
# plt.figure(5)
# plt.hist(sigmoid_detected_predicted_values, bins=15, label="true positives",
#          fc=(0, 0, 1, 0.5))
# plt.hist(sigmoid_undetected_predicted_values, bins=15, label="false positives",
#          fc=(1, 0, 0, 0.5))
# plt.xlabel("sigmoid(score value)")
# plt.ylabel("frequency")
# plt.title(str(len(predicted_coordinates)) + " peaks, threshold on score = "
#           + str(tight_threshold))
#
# plt.legend()
# plt.gcf()
# fig_name = join(figures_dir,
#                 "histogram-sigmoid-detected-undetected_thr_testset.png")
# plt.savefig(fname=fig_name,
#             format="png")

plt.figure(6)
pr_legend_str = "auPRC = " + str(round(auPRC, 4))
f1_legend_str = "(max_F1, best_peaks) = (" + str(round(max_F1, 4)) + ", " + str(
    optimal_peak_number) + ")"
title_str = str(len(predicted_coordinates_test)) + " peaks"
plt.plot(F1_score, label=f1_legend_str)
plt.xlabel("number of peaks")
plt.ylabel("F1 score")
plt.title(title_str)
plt.legend()
plt.gcf()
fig_name = join(figures_dir, "f1_score_" + title_str + ".png")
plt.savefig(fname=fig_name, format="png")

plt.figure(7)
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

stats_df = pd.read_csv(statistics_file)
stats_df['tomo_name'] = stats_df['tomo_name'].astype(str)
stats_df.loc[stats_df['tomo_name'] == tomo_name, label_name] = auPRC
stats_df.to_csv(path_or_buf=statistics_file, index=False)

