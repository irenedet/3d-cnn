import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
from os.path import join
from os import makedirs

from src.python.calculator.statistics import \
    precision_recall_calculator_and_detected, \
    pr_auc_score, \
    F1_score_calculator
from src.python.coordinates_toolbox.utils import \
    extract_coordinates_from_em_motl
from src.python.filereaders.csv import read_motl_from_csv
from src.python.filereaders.em import load_em_motl
from src.python.peak_toolbox.utils import \
    extract_motl_coordinates_and_score_values

parser = argparse.ArgumentParser()

parser.add_argument("-output", "--output_dir",
                    help="directory where the outputs will be stored",
                    type=str)
parser.add_argument("-motl", "--path_to_motl",
                    help="path to the motive list in csv format",
                    type=str)
parser.add_argument("-clean", "--path_to_clean",
                    help="path to the motive list of true particles in em format",
                    type=str)
parser.add_argument("-label", "--label_name",
                    help="name of segmentation",
                    type=str)

args = parser.parse_args()
output_dir = args.output_dir
path_to_csv_motl = args.path_to_motl
path_to_motl_clean = args.path_to_clean
label_name = args.label_name

# label_name = "ribosomes"
# # output_dir = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_005/confs_16_5_bis_/"
# # path_to_csv_motl = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_005/confs_16_5_bis_/motl_4654.csv"
#
# output_dir = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_006/confs_4_5_bis_/"
# path_to_csv_motl = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_006/confs_4_5_bis_/motl_4896.csv"
#
#
#
# # path_to_motl_clean = '/scratch/trueba/cnn/004/4bin/cnn/motl_clean_4b.em'
# # path_to_motl_clean = '/scratch/trueba/3d-cnn/clean/180426_005/motl_clean_4b.em'
# path_to_motl_clean = '/scratch/trueba/3d-cnn/clean/180426_006/motl_clean_4b.em'
figures_dir = join(output_dir, "figures")
makedirs(name=figures_dir, exist_ok=True)
# Extract coordinates from template matching
Header, motl_true = load_em_motl(path_to_emfile=path_to_motl_clean)
motl_clean_coords = extract_coordinates_from_em_motl(motl_true)
# motl_clean_coords[:, 0] += -16*np.ones(motl_clean_coords.shape[0])
# Extract coordinates from the UNet segmentation:
motl_predicted = read_motl_from_csv(path_to_csv_motl)
motl_values, motl_coordinates = extract_motl_coordinates_and_score_values(
    motl_predicted)
del motl_predicted
# Adjust to original tomogram dimensions:
# motl_coordinates = [point + np.array([0, 0, 330]) for point in motl_coordinates]

precision, recall, detected_true, detected_predicted, undetected_predicted, \
value_detected_predicted, value_undetected_predicted = \
    precision_recall_calculator_and_detected(
        motl_coordinates,
        motl_values,
        motl_clean_coords,
        radius=8)

matplotlib.use('Agg')
plt.ioff()
plt.figure(1)
plt.hist(motl_values, bins=50, label="Predicted particles")
plt.xlabel("score value")
plt.ylabel("frequency")
plt.title(str(len(motl_coordinates)) + " peaks, " + label_name)
plt.legend()
plt.gcf()
figure_name = join(figures_dir, "histogram180426_004.png")
plt.savefig(fname=figure_name,
            format="png")

matplotlib.use('Agg')
plt.ioff()
plt.figure(2)
plt.hist(value_detected_predicted, bins=70, label="true positives",
         fc=(0, 0, 1, 0.5))
plt.hist(value_undetected_predicted, bins=70, label="false positives",
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
