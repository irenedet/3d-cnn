import matplotlib.pyplot as plt
import numpy as np

from src.python.calculator.statistics import \
    precision_recall_calculator_and_detected, \
    pr_auc_score, \
    F1_score_calculator
from src.python.coordinates_toolbox.utils import \
    extract_coordinates_from_em_motl
from src.python.filereaders.csv import read_motl_from_csv
from src.python.filereaders.em import load_motl
from peak_toolbox.utils import extract_motl_coordinates_and_score_values

# filtered 50 per tomo - is the best!:
path_to_csv_motl = "/scratch/trueba/3d-cnn/TEST/motl_unique/motl_4444.csv"
path_to_motl_clean = '/home/papalotl/Sara_Goetz/180426/004/motl_clean_4b.em'

# Extract coordinates from template matching
Header, motl_clean = load_motl(path_to_emfile=path_to_motl_clean)
motl_clean_coords = extract_coordinates_from_em_motl(motl_clean)

# Extract coordinates from the UNet segmentation:
motl_ribos = read_motl_from_csv(path_to_csv_motl)
motl_values, motl_coordinates = extract_motl_coordinates_and_score_values(
    motl_ribos)
del motl_ribos
# Adjust to original tomogram dimensions:
motl_coordinates = [point + np.array([0, 0, 380]) for point in motl_coordinates]

precision, recall, detected_true, detected_predicted, undetected_predicted, \
value_detected_predicted, value_undetected_predicted = \
    precision_recall_calculator_and_detected(
        motl_coordinates,
        motl_values,
        motl_clean_coords,
        radius=8)

print("The detected fraction of positives is ",
      len(detected_true) / len(motl_clean_coords))
plt.hist(motl_values, bins=50, label="Predicted particles")
plt.xlabel("score value")
plt.ylabel("frequency")
plt.title(str(len(motl_coordinates)) + " peaks, ribosomes in 180426/004")
plt.legend()
plt.gcf()
plt.savefig(fname="/home/papalotl/Desktop/histogram180426_004.png",
            format="png")
plt.show()

plt.hist(value_detected_predicted, bins=70, label="true positives",
         fc=(0, 0, 1, 0.5))
plt.hist(value_undetected_predicted, bins=70, label="false positives",
         fc=(1, 0, 0, 0.5))
plt.xlabel("score value")
plt.ylabel("frequency")
plt.title(str(len(motl_coordinates)) + " peaks, ribosomes in 180426/004")
plt.legend()
plt.gcf()
plt.savefig(fname="/home/papalotl/Desktop/histogram-detected-undetected.png",
            format="png")
plt.show()

# plt.hist(value_detected_predicted, bins=30, label="true positives")
# plt.xlabel("score value")
# plt.ylabel("frequency")
# plt.title(str(len(motl_coords)) + " peaks, ribosomes in 180426/004")
# plt.legend()
# plt.gcf()
# # plt.savefig(fname="/scratch/trueba/3d-cnn/TEST/histogram180426_004.png",
# #             format="png")
# plt.show()
#
# plt.hist(value_undetected_predicted, bins=30, label="false positives")
# plt.xlabel("score value")
# plt.ylabel("frequency")
# plt.title(str(len(motl_coords)) + " peaks, ribosomes in 180426/004")
# plt.legend()
# plt.gcf()
# # plt.savefig(fname="/scratch/trueba/3d-cnn/TEST/histogram180426_004.png",
# #             format="png")
# plt.show()
F1_score = F1_score_calculator(precision, recall)
max_F1 = np.max(F1_score)
optimal_peak_number = np.min(np.where(F1_score == max_F1)[0])
auPRC = pr_auc_score(precision=precision, recall=recall)

pr_legend_str = "RIBOSOMES 004, auPRC = " + str(round(auPRC, 4))
f1_legend_str = "RIBOSOMES 004, (max_F1, best_peaks) = (" + str(
    round(max_F1, 4)) + ", " + str(
    optimal_peak_number) + ")"

plt.plot(F1_score, label=f1_legend_str)
plt.xlabel("number of peaks")
plt.ylabel("F1 score")
plt.title("3D UNet, 6 layers, 128^3 voxel training set")
plt.legend()
plt.gcf()
plt.savefig(fname="/home/papalotl/Desktop/F1_score_with_overlap.png",
            format="png")
plt.show()

plt.plot(recall, precision, label=pr_legend_str)
plt.xlabel("recall")
plt.ylabel("precision")
plt.title("3D UNet, 6 layers, 128^3 voxel training set")
plt.legend()
plt.gcf()
plt.savefig(fname="/home/papalotl/Desktop/PR_curve_with_overlap.png",
            format="png")
plt.show()
