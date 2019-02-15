import numpy as np
import argparse

from src.python.filewriters.csv import \
    write_global_motl_from_overlapping_subtomograms

# from os.path import join

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
parser.add_argument("-box", "--box_side",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-zdim", "--output_zdim",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-overlap", "--overlap",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-min_peak_distance", "--min_peak_distance",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-z_shift", "--z_shift_original",
                    help="name of category to be segmented",
                    type=int)

args = parser.parse_args()
output_dir = args.output_dir
label_name = args.label_name
subtomo_path = args.subtomo_path
box_side = args.box_side
output_zdim = args.output_zdim
overlap = args.overlap
min_peak_distance = args.min_peak_distance
z_shift = args.z_shift_original


subtomo_shape = tuple(box_side * np.array([1, 1, 1]))
output_shape = (output_zdim, 927, 927)
#
# subtomo_path = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_005/confs_16_5_bis_/4bin_subtomograms_.h5"
# label_name = "ribosomes"
# output_dir = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_005/confs_16_5_bis_/"
# subtomo_shape = (128, 128, 128)
# output_shape = (251, 927, 927)
# min_peak_distance = 12
# overlap = 12
# z_shift = 370

# subtomo_path = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_006/confs_8_5_bis_/4bin_subtomograms_.h5"
# label_name = "ribosomes"
# output_dir = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_006/confs_8_5_bis_/"
# subtomo_shape = (128, 128, 128)
# output_shape = (321, 927, 927)
# min_peak_distance = 12
# overlap = 12
# z_shift = 0#370

#
# Future local parameters:
subtomo_peaks_number = 40
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
    number_peaks_uniquify=number_peaks_uniquify,
    z_shift=z_shift)

print(motl_file_name)
