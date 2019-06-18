import sys
sys.path.insert(0, '/g/scb2/zaugg/trueba/3d-cnn/src')
print(sys.path)
import numpy as np
import argparse
from os import makedirs

from src.python.filewriters.csv import \
    write_global_motl_from_overlapping_subtomograms_multiclass

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
parser.add_argument("-xdim", "--output_xdim",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-ydim", "--output_ydim",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-zdim", "--output_zdim",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-overlap", "--overlap",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-class_number", "--class_number",
                    help="class number associated to motifs list",
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
output_xdim = args.output_xdim
output_ydim = args.output_ydim
output_zdim = args.output_zdim
overlap = args.overlap
class_number = args.class_number
min_peak_distance = args.min_peak_distance
z_shift = args.z_shift_original


subtomo_shape = tuple(box_side * np.array([1, 1, 1]))
output_shape = (output_zdim, output_ydim, output_xdim)
makedirs(name=output_dir, exist_ok=True)

# Todo? Future local parameters:
peaks_per_subtomo = int(box_side ** 3 / (2 * min_peak_distance) ** 3)
print("peaks per subtomo = ", peaks_per_subtomo)
number_peaks_uniquify = 7000

# subtomo_shape = tuple(box_side * np.array([1, 1, 1]))
# output_shape = (output_zdim, output_ydim, output_xdim)
# makedirs(name=output_dir, exist_ok=True)
# # Future local parameters:
# peaks_per_subtomo = 40
# number_peaks_uniquify = 7000

motl_file_name = write_global_motl_from_overlapping_subtomograms_multiclass(
    subtomograms_path=subtomo_path,
    motive_list_output_dir=output_dir,
    overlap=overlap,
    label_name=label_name,
    output_shape=output_shape,
    subtomo_shape=subtomo_shape,
    numb_peaks=peaks_per_subtomo,
    class_number=class_number,
    min_peak_distance=min_peak_distance,
    number_peaks_uniquify=number_peaks_uniquify,
    z_shift=z_shift)

print(motl_file_name)
