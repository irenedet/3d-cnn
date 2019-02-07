from src.python.filewriters.csv import \
    write_global_motl_from_overlapping_subtomograms
import argparse
from os.path import join

parser = argparse.ArgumentParser()

parser.add_argument("-output", "--output_dir",
                    help="directory where the outputs will be stored",
                    type=str)
parser.add_argument("-label", "--label_name",
                    help="name of category to be segmented",
                    type=str)
parser.add_argument("-subtomo", "--subtomo_path",
                    help="name of category to be segmented",
                    type=str)

args = parser.parse_args()
output_dir = args.output_dir
label_name = args.label_name
subtomo_path = args.subtomo_path

subtomo_shape = (128, 128, 128)
output_shape = (221, 928, 928)
min_peak_distance = 12
overlap = 12

# Future local parameters:
subtomo_peaks_number = 50
number_peaks_uniquify = 7000

motl_file_name = write_global_motl_from_overlapping_subtomograms(
    subtomograms_path=subtomo_path,
    motive_list_output_dir=output_dir,
    overlap=overlap,
    label_name=label_name,
    output_shape=output_shape,
    subtomo_shape=subtomo_shape,
    numb_peaks=subtomo_peaks_number,
    min_peak_distance=min_peak_distance,
    number_peaks_uniquify=number_peaks_uniquify)

exit(motl_file_name)

