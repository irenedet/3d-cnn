import time

import matplotlib.pyplot as plt
import numpy as np

from performance.statistics_utils import pr_auc_score, \
    f1_score_calculator, precision_recall_calculator
from coordinates_toolbox.utils import \
    extract_coordinates_from_em_motl
from file_actions.readers.motl import read_motl_from_csv
from file_actions.readers.em import read_em
from peak_toolbox.subtomos import select_coordinates_in_subtomos, \
    select_coordinates_and_values_in_subtomos

path_to_csv_motl = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_004/gaussian_aug/confs_4_5_/motl_4470.csv"
# path_to_csv_motl = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_004/confs_4_5_bis_/motl_4623.csv"
# "/scratch/trueba/3d-cnn/cnn_evaluation/180426_004/motl_4431.csv"#"/scratch/trueba/3d-cnn/TEST/motl_unique/motl_4444.csv"
path_to_motl_clean = '/home/papalotl/Sara_Goetz/180426/004/motl_clean_4b.em'

motl_predicted = read_motl_from_csv(path_to_csv_motl)
Header, motl_true = read_em(path_to_emfile=path_to_motl_clean)

true_coordinates = extract_coordinates_from_em_motl(motl_true)
print("len(true_coordinates) = ", len(true_coordinates))

peaks_number = 6000
predicted_coordinates = [np.array([row[7], row[8], row[9]]) for row in
                         motl_predicted[:peaks_number]]
motl_values = [row[0] for row in motl_predicted[:peaks_number]]
del motl_predicted
unique_peaks_number = len(motl_values)
# from coordinates_toolbox.utils import filtering_duplicate_coords_with_values

training_data_path = \
    "/scratch/trueba/3d-cnn/training_data/TEST/ribo_training.h5"
# '/scratch/trueba/3d-cnn/training_data/training_data_side128_49examples.h5'
data_order_while_training = \
    [304, 175, 180, 86, 9, 45, 240, 33, 220, 338, 151, 252, 328, 6, 343, 283,
     147, 163, 388, 329, 31, 285, 14, 299, 271, 376, 171, 65, 268, 121, 331,
     235, 170, 341, 370, 307, 19, 221, 360, 92, 74, 52, 72, 277, 82, 196, 46,
     100, 88, 114, 108, 135, 49, 89, 273, 280, 292, 287, 211, 272, 28, 373, 313,
     34, 32, 309, 236, 378, 390, 90, 229, 209, 332, 159, 160, 187, 226, 87, 79,
     336, 51, 294, 182, 126, 13, 339, 282, 47, 3, 337, 102, 73, 167, 367, 311,
     305, 77, 369, 233, 206, 35, 119, 212, 189, 225, 230, 165, 381, 205, 93,
     134, 104, 312, 155, 293, 315, 242, 375, 2, 66, 0, 83, 276, 266, 290, 40,
     351, 85, 198, 256, 11, 385, 325, 186, 245, 111, 237, 244, 259, 58, 202,
     248, 207, 42, 327, 352, 176, 146, 278, 48, 265, 7, 106, 232, 347, 59, 359,
     382, 258, 129, 162, 20, 22, 301, 364, 67, 25, 241, 173, 161, 130, 286, 295,
     142, 4, 200, 319, 316, 371, 201, 27, 210, 353, 125, 112, 192, 213, 164,
     261, 71, 318, 348, 110, 345, 255, 257, 68, 274, 308, 335, 310, 177, 50, 53,
     322, 183, 222, 156, 368, 62, 184, 208, 320, 70, 379, 342, 36, 94, 247, 227,
     136, 181, 37, 270, 223, 118, 149, 101, 12, 109, 298, 168, 362, 16, 84, 314,
     193, 321, 334, 30, 55, 18, 139, 269, 188, 297, 306, 377, 264, 185, 124,
     152, 300, 238, 131, 57, 190, 132, 204, 387, 63, 216, 133, 140, 154, 76,
     138, 24, 275, 281, 141, 224, 153, 178, 194, 144, 372, 284, 15, 113, 366,
     253, 344, 228, 78, 358, 380, 346, 172, 251, 21, 95, 115, 246, 105, 239,
     231, 128, 363, 288, 60, 279, 324, 117, 389, 157, 386, 145, 234, 122, 215,
     123, 97, 361, 383, 291, 120, 249, 17, 148, 43, 365, 44, 56, 356, 103, 69,
     61, 26, 81, 195, 350, 8, 1, 354, 169, 296, 10, 116, 355, 217, 99, 303, 262,
     166, 263, 127, 158, 39, 179, 218, 174, 326, 75, 41, 323, 98, 333, 349, 80,
     38, 29, 384, 5, 150, 243, 317, 250, 219, 143, 203, 260, 91, 330, 214, 199,
     54, 64, 357, 254, 289, 197, 107, 340, 302, 137, 267, 374, 23, 96, 191]

training_set_size = len(data_order_while_training)
data_order_while_training = [number for number in data_order_while_training
                             if number <= training_set_size // 2]
# data_order_while_training = [35, 24, 15, 31, 2, 20, 41, 7, 34, 42, 26, 5, 47,
#                              23, 33, 0, 16, 29, 21, 12, 17, 8, 4, 27, 3, 40, 32,
#                              38, 37, 39, 6, 11, 9, 46, 22, 28, 18, 1, 36, 25,
#                              13, 14, 48, 44, 10, 30, 43, 19,
#                              45]  # list(range(49))

# data_order = [48, 20, 27, 19, 17, 44, 35, 40, 31, 15, 43, 33, 26, 25, 29, 8, 28,
# 45, 47, 36, 41, 5, 21, 37, 16, 18, 46, 42, 11, 13, 3, 12, 30, 1,
# 2, 0, 22, 9, 34, 10, 6, 7, 24, 38, 23, 39, 14, 4, 32]
training_split = 170
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
        shift=shift)
end = time.time()
print("elapsed time for splitting test and train coordinates", end - start,
      "sec")
print("predicted_coordinates_test", len(predicted_coordinates_test))
shift = 0
predicted_coordinates_test = [np.array(p) + np.array([0, 0, shift]) for p in
                              predicted_coordinates_test]
predicted_coordinates_test = np.array(predicted_coordinates_test)

print(len(predicted_coordinates_test))

precision, recall, detected_clean, detected_predicted, \
undetected_predicted, value_detected_predicted, value_undetected_predicted = \
    precision_recall_calculator(
        predicted_coordinates_test,
        predicted_coordinates_test_values,
        true_coordinates_test,
        radius=8)
print("len(predicted_coordinates_test_values)",
      len(predicted_coordinates_test_values))

indices = np.where(np.array(predicted_coordinates_test_values) > 0.1)[0]
predicted_coordinates_test_values_range = [
    predicted_coordinates_test_values[index] for index in
    indices]
plt.hist(predicted_coordinates_test_values_range, bins=10, label="all detected")
plt.xlabel("score value")
plt.ylabel("frequency")
plt.title(str(len(predicted_coordinates)) + " peaks, ribosomes in 180426/004")
plt.legend()
plt.gcf()
plt.savefig(fname="/home/papalotl/Desktop/histogram180426_004_testset.png",
            format="png")
plt.show()
sigmoid = lambda t: 1 / (1 + np.exp(-t))

sigmoid_predicted_coordinates_test_values = [sigmoid(t) for t in
                                             predicted_coordinates_test_values_range]

plt.hist(sigmoid_predicted_coordinates_test_values, bins=10,
         label="all detected")
plt.xlabel("sigmoid(score value)")
plt.ylabel("frequency")
plt.title(str(len(predicted_coordinates)) + " peaks, ribosomes in 180426/004")
plt.legend()
plt.gcf()
plt.savefig(fname="/home/papalotl/Desktop/histogram_sigmoid_004_testset.png",
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

sigmoid_detected_predicted_values = [sigmoid(value) for value in
                                     value_detected_predicted_range]
sigmoid_undetected_predicted_values = [sigmoid(value) for value in
                                       value_undetected_predicted_range]

plt.hist(sigmoid_detected_predicted_values, bins=15, label="true positives",
         fc=(0, 0, 1, 0.5))
plt.hist(sigmoid_undetected_predicted_values,
         bins=15, label="false positives",
         fc=(1, 0, 0, 0.5))
plt.xlabel("sigmoid(score value)")
plt.ylabel("frequency")
plt.title(str(len(predicted_coordinates)) + " peaks, ribosomes in 180426/004")
plt.legend()
plt.gcf()
plt.savefig(
    fname="/home/papalotl/Desktop/histogram-sigmoid-detected-undetected_testset.png",
    format="png")
plt.show()

F1_score = f1_score_calculator(precision, recall)
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
plt.savefig(fname="/home/papalotl/Desktop/f1_score_" + title_str + ".png",
            format="png")
plt.show()

plt.plot(recall, precision, label=pr_legend_str)
plt.xlabel("recall")
plt.ylabel("precision")
plt.title(title_str)
plt.legend()
plt.gcf()
plt.savefig(fname="/home/papalotl/Desktop/pr_" + title_str + ".png",
            format="png")
plt.show()
