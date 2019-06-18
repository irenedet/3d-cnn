from src.python.calculator.statistics import precision_recall_calculator, \
    pr_auc_score, \
    F1_score_calculator
from src.python.coordinates_toolbox.utils import extract_coordinates_from_em_motl
from src.python.filereaders.csv import read_motl_from_csv
from src.python.filereaders.em import read_em
import matplotlib.pyplot as plt
import numpy as np

path_to_csv_motl = "/scratch/trueba/3d-cnn/TEST/motl_18019.csv"
# path_to_csv_motl = "/scratch/trueba/3d-cnn/TEST/motl_32367.csv"#wrong: but as expected: applied old tools to overlapped subtomos
# path_to_csv_motl = "/scratch/trueba/3d-cnn/TEST/motl_19254.csv"#good! without overlap
# path_to_csv_motl = "/scratch/trueba/3d-cnn/TEST/motl_19279.csv"#wrong!
# path_to_csv_motl = "/scratch/trueba/3d-cnn/TEST/motl_10355.csv"
path_to_csv_motl2 = '/scratch/trueba/3d-cnn/TEST/motl_245558.csv'  # good! original w.o. overlap
# path_to_motl_clean = '/home/papalotl/Sara_Goetz/180426/004/motl_clean_4b.em'

motl_ribos = read_motl_from_csv(path_to_csv_motl)
motl_ribos_wo_overlap = read_motl_from_csv(path_to_csv_motl2)
# Header, motl_clean = read_em(path_to_emfile=path_to_motl_clean)

# motl_clean_coords = extract_coordinates_from_em_motl(motl_clean)
motl_coords = [tuple([row[7], row[8], row[9] + 380]) for row in
               motl_ribos[:3000]]
motl_coords_wo_overlap = np.array([tuple([row[7], row[8], row[9] + 380]) for row in
                          motl_ribos_wo_overlap[:3000]])

# assert len(motl_coords) == len(set(motl_coords))

precision, recall, detected_clean = precision_recall_calculator(motl_coords,
                                                                motl_coords_wo_overlap,
                                                                radius=8)

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
plt.show()

plt.plot(recall, precision, label=pr_legend_str)
plt.xlabel("recall")
plt.ylabel("precision")
plt.title("3D UNet, 6 layers, 128^3 voxel training set")
plt.legend()
plt.show()
