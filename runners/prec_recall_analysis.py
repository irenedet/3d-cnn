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

label_name = "ribosomes"
# path_to_csv_motl = "/home/papalotl/Sara_Goetz/180426/005/motl/motl_3000.csv"
# path_to_motl_clean = '/home/papalotl/Sara_Goetz/180426/005/motl/motl_clean_4b.em'
# output_dir = "/home/papalotl/Sara_Goetz/180426/005/_2000_peaks"

# path_to_csv_motl = "/home/papalotl/Sara_Goetz/180426/006/motl_4004.csv"
# path_to_motl_clean = '/home/papalotl/Sara_Goetz/180426/006/motl/motl_clean_4b.em'
# output_dir = "/home/papalotl/Sara_Goetz/180426/006"

# output_dir = "/home/papalotl/Sara_Goetz/180426/004/cnn_sph_train/23postrain/motl/"
# path_to_csv_motl = "/home/papalotl/Sara_Goetz/180426/004/cnn_sph_train/23postrain/motl/motl_4000.csv"
# path_to_motl_clean = '/home/papalotl/Sara_Goetz/180426/004/motl_clean_4b.em'
# output_dir = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_005/confs_16_5_bis_/"
# path_to_csv_motl = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_005/confs_16_5_bis_/motl_4654.csv"

# path_to_csv_motl = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_005/confs_4_5_bis_/motl_4662.csv"
# output_dir = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_006/confs_4_5_bis_"
output_dir = "/home/papalotl/006_confs_4_5_bis_"
path_to_csv_motl = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_006/confs_4_5_bis_/motl_4896.csv"

# path_to_csv_motl = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_005/confs_4_5_bis_/motl_4896.csv"


# path_to_csv_motl = \
# "/scratch/trueba/3d-cnn/cnn_evaluation/180426_006/from_004_training/gaussian_aug/confs_4_5_/motl_4810.csv"
#"/scratch/trueba/3d-cnn/cnn_evaluation/180426_004/gaussian_aug/confs_4_5_/motl_4470.csv"
# "/scratch/trueba/3d-cnn/cnn_evaluation/180426_004/motl_4431.csv"#"/scratch/trueba/3d-cnn/TEST/motl_unique/motl_4444.csv"
# path_to_motl_clean = '/scratch/trueba/cnn/004/4bin/cnn/motl_clean_4b.em'

# output_dir = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_004/gaussian_aug/confs_4_5_/"
# "/scratch/trueba/3d-cnn/cnn_evaluation/180426_006/from_004_training/gaussian_aug/confs_4_5_/"
# "/scratch/trueba/3d-cnn/cnn_evaluation/180426_005/from_004_training/gaussian_aug/confs_4_5_/_peaks_test_and_training_set/"
#
# path_to_motl_clean = '/scratch/trueba/cnn/004/4bin/cnn/motl_clean_4b.em'
# path_to_motl_clean = '/scratch/trueba/3d-cnn/clean/180426_005/motl_clean_4b.em'
path_to_motl_clean = '/scratch/trueba/3d-cnn/clean/180426_006/motl_clean_4b.em'
figures_dir = join(output_dir, "figures")
makedirs(name=figures_dir, exist_ok=True)
# Extract coordinates from template matching
Header, motl_true = load_em_motl(path_to_emfile=path_to_motl_clean)
motl_clean_coords = extract_coordinates_from_em_motl(motl_true)
# motl_clean_coords[:, 0] += -16*np.ones(motl_clean_coords.shape[0])
# motl_clean_coords[:, 2] += -40*np.ones(motl_clean_coords.shape[0])
# Extract coordinates from the UNet segmentation:
motl_predicted = read_motl_from_csv(path_to_csv_motl)
motl_values, motl_coordinates = extract_motl_coordinates_and_score_values(
    motl_predicted)
del motl_predicted

sigmoid = lambda t: 1 / (1 + np.exp(-t))

sigmoid_motl_values = [sigmoid(value) for value in motl_values if value > 0]

matplotlib.use('Agg')
plt.ioff()
plt.figure(7)
plt.hist(sigmoid_motl_values, bins=15, label="Predicted particles")
plt.xlabel("sigmoid(score value)")
plt.ylabel("frequency")
plt.title(str(len(motl_coordinates)) + " peaks, " + label_name)
plt.legend()
plt.gcf()
figure_name = join(figures_dir, "histogram_sigmoid_all_values.png")
plt.savefig(fname=figure_name,
            format="png")
# motl_values /= np.max(motl_values)
n = 5000
print("len(motl_clean_coords)", n)
motl_values, motl_coordinates = motl_values[:n], motl_coordinates[:n]
motl_values = np.array(motl_values)
# For old motl:
# motl_coordinates = [[row[1]+16, row[0], row[2] + 370] for row in motl_coordinates]

# Adjust to original tomogram dimensions:
# motl_coordinates = [point + np.array([16, 0, 0]) for point in
#                     motl_coordinates]  # for 005

precision, recall, detected_true, detected_predicted, undetected_predicted, \
value_detected_predicted, value_undetected_predicted = \
    precision_recall_calculator_and_detected(
        motl_coordinates,
        motl_values,
        motl_clean_coords,
        radius=8)

detected_predicted = [np.array(point) for point in detected_predicted]
from src.python.filewriters.csv import unique_coordinates_motl_writer

unique_coordinates_motl_writer(path_to_output_folder=output_dir,
                               list_of_peak_scores=value_detected_predicted,
                               list_of_peak_coords=detected_predicted,
                               number_peaks_to_uniquify=5000,
                               minimum_peaks_distance=12)

sigmoid_value_detected_predicted = [sigmoid(value) for value in
                                    value_detected_predicted]
sigmoid_value_undetected_predicted = [sigmoid(value) for value in
                                      value_undetected_predicted]
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
plt.figure(6)
plt.hist(sigmoid_motl_values, bins=30, label="Predicted particles")
plt.xlabel("sigmoid(score value)")
plt.ylabel("frequency")
plt.title(str(len(motl_coordinates)) + " peaks, " + label_name)
plt.legend()
plt.gcf()
figure_name = join(figures_dir, "histogram_sigmoid.png")
plt.savefig(fname=figure_name,
            format="png")

matplotlib.use('Agg')
plt.ioff()
plt.figure(5)
plt.hist(sigmoid_value_detected_predicted, bins=25, label="true positives",
         fc=(0, 0, 1, 0.5))
plt.hist(sigmoid_value_undetected_predicted, bins=25, label="false positives",
         fc=(1, 0, 0, 0.5))
plt.xlabel("score value")
plt.ylabel("frequency")
plt.title(str(len(motl_coordinates)) + " peaks, " + label_name)
plt.legend()
plt.gcf()
figure_name = join(figures_dir, "sigmoid_histogram-detected-undetected.png")
plt.savefig(fname=figure_name,
            format="png")

matplotlib.use('Agg')
plt.ioff()
plt.figure(2)
plt.hist(value_detected_predicted, bins=25, label="true positives",
         fc=(0, 0, 1, 0.5))
plt.hist(value_undetected_predicted, bins=25, label="false positives",
         fc=(1, 0, 0, 0.5))
plt.xlabel("sigmoid(score value)")
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
