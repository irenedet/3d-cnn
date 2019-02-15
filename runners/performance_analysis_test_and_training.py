from src.python.calculator.statistics import precision_recall_calculator, \
    pr_auc_score, \
    F1_score_calculator, precision_recall_calculator_and_detected
from src.python.coordinates_toolbox.utils import \
    extract_coordinates_from_em_motl
from src.python.filereaders.csv import read_motl_from_csv
from src.python.filereaders.em import load_em_motl
import matplotlib.pyplot as plt
import numpy as np


def _filter_coordinates_per_subtomo(coordinates, subtomo_corners_init,
                                    subtomo_corners_end, shift_z):
    # print("subtomo_corners_init", subtomo_corners_init)
    # print("subtomo_corners_end", subtomo_corners_end)
    subtomo_corners_init_xyz = list(reversed(subtomo_corners_init))
    subtomo_corners_end_xyz = list(reversed(subtomo_corners_end))
    subtomo_corners_init_xyz = np.array(subtomo_corners_init_xyz) + np.array(
        [0, 0, shift_z])
    subtomo_corners_end_xyz = np.array(subtomo_corners_end_xyz) + np.array(
        [0, 0, shift_z])  # hasta aca bien!

    # print("subtomo", subtomo_corners_init_xyz, subtomo_corners_end_xyz)
    selected_coordinates = []
    discarded_coordinates = []
    for point in coordinates:
        is_in_subtomo = all(p >= c_init and p <= c_end for p, c_init, c_end in
                            zip(point, subtomo_corners_init_xyz,
                                subtomo_corners_end_xyz))
        if is_in_subtomo:
            selected_coordinates += [point]
        else:
            discarded_coordinates += [point]
            print("discarded!")
    return selected_coordinates, discarded_coordinates


import h5py
from src.python.naming import h5_internal_paths


def read_subtomo_names(subtomo_file_path):
    with h5py.File(subtomo_file_path, 'r') as f:
        return list(f[h5_internal_paths.RAW_SUBTOMOGRAMS])


from src.python.coordinates_toolbox.subtomos import get_coord_from_name, \
    get_subtomo_corners


def filter_test_coordinates(coordinates: list, subtomo_file_path: str,
                            split: int, dataset_shape: tuple,
                            subtomo_shape: tuple, shift: int):
    subtomo_names = read_subtomo_names(subtomo_file_path)
    test_coordinates = []
    train_coordinates = []
    for subtomo_name in subtomo_names[split:]:
        subtomo_center = get_coord_from_name(subtomo_name)
        subtomo_corners_init, subtomo_corners_end, _ = get_subtomo_corners(
            output_shape=dataset_shape, subtomo_shape=subtomo_shape,
            subtomo_center=subtomo_center)
        filtered_coordinates, discarded_coordinates = \
            _filter_coordinates_per_subtomo(coordinates,
                                            subtomo_corners_init,
                                            subtomo_corners_end,
                                            shift)
        test_coordinates += filtered_coordinates
        train_coordinates += discarded_coordinates
    return test_coordinates, train_coordinates


def _get_subtomo_coorners(subtomo_corners_init: list, subtomo_corners_end: list,
                          shift_z: int):
    subtomo_corners_init_xyz = list(reversed(subtomo_corners_init))
    subtomo_corners_end_xyz = list(reversed(subtomo_corners_end))
    subtomo_corners_init_xyz = np.array(subtomo_corners_init_xyz) + np.array(
        [0, 0, shift_z])
    subtomo_corners_end_xyz = np.array(subtomo_corners_end_xyz) + np.array(
        [0, 0, shift_z])
    return subtomo_corners_init_xyz, subtomo_corners_end_xyz


def _get_subtomo_corners_from_split(subtomo_names: list, split: int,
                                    subtomo_shape: tuple, dataset_shape: tuple,
                                    shift: int):
    subtomos_corners = []
    for subtomo_name in subtomo_names[split:]:
        subtomo_center = get_coord_from_name(subtomo_name)
        subtomo_corners_init, subtomo_corners_end, _ = get_subtomo_corners(
            output_shape=dataset_shape, subtomo_shape=subtomo_shape,
            subtomo_center=subtomo_center)
        corners = _get_subtomo_coorners(subtomo_corners_init,
                                        subtomo_corners_end, shift)
        subtomos_corners += [corners]
    return subtomos_corners


def _check_point_in_subtomo(corners: tuple, point: list):
    subtomo_corners_init_xyz, subtomo_corners_end_xyz = corners
    is_in_subtomo = all(
        p >= c_init and p <= c_end for p, c_init, c_end in
        zip(point, subtomo_corners_init_xyz,
            subtomo_corners_end_xyz))
    return is_in_subtomo


def select_coordinates_in_subtomos(coordinates: list, subtomo_file_path: str,
                                   split: int,
                                   data_order: list, dataset_shape: tuple,
                                   subtomo_shape: tuple,
                                   shift: int):
    subtomo_names = read_subtomo_names(subtomo_file_path)
    subtomo_names_reordered = [subtomo_names[i] for i in data_order]
    subtomos_corners = _get_subtomo_corners_from_split(subtomo_names_reordered,
                                                       split,
                                                       subtomo_shape,
                                                       dataset_shape, shift)
    selected_coordinates = []
    discarded_coordinates = []
    for point in coordinates:
        for corners in subtomos_corners:
            is_in_subtomo = _check_point_in_subtomo(corners, point)
            if is_in_subtomo:
                selected_coordinates += [point]
            else:
                discarded_coordinates += [point]
    return selected_coordinates, discarded_coordinates


def select_coordinates_and_values_in_subtomos(coordinates: list,
                                              values: list,
                                              subtomo_file_path: str,
                                              split: int,
                                              data_order: list,
                                              dataset_shape: tuple,
                                              subtomo_shape: tuple,
                                              shift: int):
    subtomo_names = read_subtomo_names(subtomo_file_path)
    subtomo_names_reordered = [subtomo_names[i] for i in data_order]
    subtomos_corners = _get_subtomo_corners_from_split(subtomo_names_reordered,
                                                       split,
                                                       subtomo_shape,
                                                       dataset_shape,
                                                       shift)
    selected_coordinates = []
    selected_coordinates_values = []
    discarded_coordinates = []
    discarded_coordinates_values = []
    for value, point in zip(values, coordinates):
        for corners in subtomos_corners:
            is_in_subtomo = _check_point_in_subtomo(corners, point)
            if is_in_subtomo:
                selected_coordinates += [point]
                selected_coordinates_values += [value]
            else:
                discarded_coordinates += [point]
                discarded_coordinates_values += [value]
    return selected_coordinates, discarded_coordinates, \
           selected_coordinates_values, discarded_coordinates_values


path_to_csv_motl = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_004/confs_16_5_bis_/motl_4569.csv"
# "/scratch/trueba/3d-cnn/cnn_evaluation/180426_004/motl_4431.csv"#"/scratch/trueba/3d-cnn/TEST/motl_unique/motl_4444.csv"
path_to_motl_clean = '/home/papalotl/Sara_Goetz/180426/004/motl_clean_4b.em'

motl_predicted = read_motl_from_csv(path_to_csv_motl)
Header, motl_true = load_em_motl(path_to_emfile=path_to_motl_clean)

true_coordinates = extract_coordinates_from_em_motl(motl_true)
print("len(true_coordinates) = ", len(true_coordinates))

peaks_number = 7000
predicted_coordinates = [np.array([row[7], row[8], row[9]]) for row in
                         motl_predicted[:peaks_number]]
motl_values = [row[0] for row in motl_predicted[:peaks_number]]
del motl_predicted
unique_peaks_number = len(motl_values)
# from coordinates_toolbox.utils import filtering_duplicate_coords_with_values
import time

training_data_path = \
    '/scratch/trueba/3d-cnn/training_data/training_data_side128_49examples.h5'
data_order_while_training = [35, 24, 15, 31, 2, 20, 41, 7, 34, 42, 26, 5, 47,
                             23, 33, 0, 16, 29, 21, 12, 17, 8, 4, 27, 3, 40, 32,
                             38, 37, 39, 6, 11, 9, 46, 22, 28, 18, 1, 36, 25,
                             13, 14, 48, 44, 10, 30, 43, 19,
                             45]  # list(range(49))
# data_order = [48, 20, 27, 19, 17, 44, 35, 40, 31, 15, 43, 33, 26, 25, 29, 8, 28,
# 45, 47, 36, 41, 5, 21, 37, 16, 18, 46, 42, 11, 13, 3, 12, 30, 1,
# 2, 0, 22, 9, 34, 10, 6, 7, 24, 38, 23, 39, 14, 4, 32]
training_split = 34
dataset_shape = (221, 928, 928)
overlap = 12
subtomo_shape = (128, 128, 128)

shift = 380
start = time.time()
true_coordinates_test, _ = \
    select_coordinates_in_subtomos(
        coordinates=true_coordinates,
        subtomo_file_path=training_data_path,
        split=training_split,
        data_order=data_order_while_training,
        dataset_shape=dataset_shape,
        subtomo_shape=subtomo_shape,
        shift=shift)
true_coordinates_test = np.array(true_coordinates_test)
end = time.time()
print("elapsed time for splitting  test and train coordinates", end - start,
      "sec")
print("true_coordinates_test", len(true_coordinates_test))

start = time.time()
predicted_coordinates_test, _, predicted_coordinates_test_values, _ = \
    select_coordinates_and_values_in_subtomos(
        coordinates=predicted_coordinates,
        values=motl_values,
        subtomo_file_path=training_data_path,
        split=training_split,
        data_order=data_order_while_training,
        dataset_shape=dataset_shape,
        subtomo_shape=subtomo_shape,
        shift=0)
end = time.time()
print("elapsed time for splitting test and train coordinates", end - start,
      "sec")
print("predicted_coordinates_test", len(predicted_coordinates_test))

predicted_coordinates_test = [np.array(p) + np.array([0, 0, shift]) for p in
                              predicted_coordinates_test]
predicted_coordinates_test = np.array(predicted_coordinates_test)

print(len(predicted_coordinates_test))

precision, recall, detected_clean, detected_predicted, \
undetected_predicted, value_detected_predicted, value_undetected_predicted = \
    precision_recall_calculator_and_detected(
        predicted_coordinates_test,
        predicted_coordinates_test_values,
        true_coordinates_test,
        radius=8)
print("len(predicted_coordinates_test_values)",
      len(predicted_coordinates_test_values))

indices = np.where(np.array(predicted_coordinates_test_values) > 0.2)[0]
predicted_coordinates_test_values_range = [
    predicted_coordinates_test_values[index] for index in
    indices]
plt.hist(predicted_coordinates_test_values_range, bins=15, label="all detected")
plt.xlabel("score value")
plt.ylabel("frequency")
plt.title(str(len(predicted_coordinates)) + " peaks, ribosomes in 180426/004")
plt.legend()
plt.gcf()
plt.savefig(fname="/home/papalotl/Desktop/histogram180426_004_testset.png",
            format="png")
plt.show()

indices = np.where(np.array(value_detected_predicted) > 0.2)[0]
value_detected_predicted_range = [value_detected_predicted[index] for index in
                                  indices]
indices = np.where(np.array(value_undetected_predicted) > 0.2)[0]
value_undetected_predicted_range = [value_undetected_predicted[index] for index
                                    in indices]

plt.hist(value_detected_predicted_range, bins=15, label="true positives",
         fc=(0, 0, 1, 0.5))
plt.hist(value_undetected_predicted_range,
         bins=15, label="false positives",
         fc=(1, 0, 0, 0.5))
plt.xlabel("score value")
plt.ylabel("frequency")
plt.title(str(len(predicted_coordinates)) + " peaks, ribosomes in 180426/004")
plt.legend()
plt.gcf()
plt.savefig(
    fname="/home/papalotl/Desktop/histogram-detected-undetected_testset.png",
    format="png")
plt.show()

F1_score = F1_score_calculator(precision, recall)
max_F1 = np.max(F1_score)
optimal_peak_number = np.min(np.where(F1_score == max_F1)[0])
auPRC = pr_auc_score(precision=precision, recall=recall)

pr_legend_str = "RIBOSOMES 004, auPRC = " + str(round(auPRC, 4))
f1_legend_str = "RIBOSOMES 004, (max_F1, best_peaks) = (" + str(
    round(max_F1, 4)) + ", " + str(
    optimal_peak_number) + ")"
title_str = "6_layers_128cube_side_" + str(peaks_number) + "_peaks"

plt.plot(F1_score, label=f1_legend_str)
plt.xlabel("number of peaks")
plt.ylabel("F1 score")
plt.title(title_str)
plt.legend()
plt.gcf()
plt.savefig(fname="/scratch/trueba/3d-cnn/TEST/f1_score_" + title_str + ".png",
            format="png")
plt.show()

plt.plot(recall, precision, label=pr_legend_str)
plt.xlabel("recall")
plt.ylabel("precision")
plt.title(title_str)
plt.legend()
plt.gcf()
plt.savefig(fname="/scratch/trueba/3d-cnn/TEST/pr_" + title_str + ".png",
            format="png")
plt.show()
