import matplotlib.pyplot as plt

from performance.statistics_utils import precision_recall_calculator_and_detected
from coordinates_toolbox.utils import extract_coordinates_from_em_motl
from file_actions.readers.csv import read_motl_from_csv
from file_actions.readers.em import read_em

path_to_csv_motl = '/home/papalotl/Sara_Goetz/180426/004/FAS/boxlength64/motl/motl_6000_emformat.csv'
path_to_motl_clean = '/home/papalotl/Sara_Goetz/180426/004/FAS/motl_clean_fas_4b.em'

motl_fas = read_motl_from_csv(path_to_csv_motl)
Header, motl_clean = read_em(path_to_emfile=path_to_motl_clean)

motl_clean_coords = extract_coordinates_from_em_motl(motl_clean)

motl_coords = [[row[7], row[8], row[9] + 380] for row in motl_fas]

precision, recall, detected_clean, *rest = \
    precision_recall_calculator_and_detected(predicted_coordinates=motl_coords,
                                             true_coordinates=motl_clean_coords,
                                             radius=8)

plt.plot(recall, precision)
plt.show()
