from os.path import join
from os import makedirs
from file_actions.writers.h5 import \
    write_dataset_from_subtomos_with_overlap_dice_multiclass
from src.python.constants import h5_internal_paths
import numpy as np


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-label_name", "--label_name",
                    help="label name of segmentation class",
                    type=str)
parser.add_argument("-subtomos_path", "--subtomos_path",
                    help="logs directory where training losses will be stored",
                    type=str)
parser.add_argument("-output_dir", "--output_dir",
                    help="model distinctive name",
                    type=str)
parser.add_argument("-box", "--box",
                    help="directory where the model will be stored",
                    type=int)
parser.add_argument("-overlap_thickness", "--overlap_thickness",
                    help="split between training and validation sets",
                    type=int)
parser.add_argument("-output_classes", "--output_classes",
                    help="number of output classes to be segmented",
                    type=int)
parser.add_argument("-output_xdim", "--output_xdim",
                    help="output_xdim dimension of output dataset",
                    type=int)
parser.add_argument("-output_ydim", "--output_ydim",
                    help="output_ydim dimension of output dataset",
                    type=int)
parser.add_argument("-output_zdim", "--output_zdim",
                    help="output_zdim dimension of output dataset",
                    type=int)
parser.add_argument("-segmentation_names", "--segmentation_names",
                    help="number of epoches for training",
                    type=str)

args = parser.parse_args()
subtomos_path = args.subtomos_path
output_dir = args.output_dir
output_xdim = args.output_xdim
output_ydim = args.output_ydim
output_zdim = args.output_zdim
box = args.box
overlap_thickness = args.overlap_thickness
label_name = args.label_name
output_classes = args.output_classes
segmentation_names = args.segmentation_names

makedirs(name=output_dir, exist_ok=True)
output_shape = (output_zdim, output_ydim, output_xdim)
subtomo_shape = box * np.array([1, 1, 1])
segmentation_names = segmentation_names.split(",")


subtomos_internal_path = join(
    h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS,
    label_name)
print("subtomos_internal_path =", subtomos_internal_path)
for n in range(output_classes):
    segmentation_name = segmentation_names[n]
    output_file = segmentation_name + ".hdf"
    output_path = join(output_dir, output_file)
    class_number = n
    write_dataset_from_subtomos_with_overlap_dice_multiclass(
        output_path,
        subtomos_path,
        output_shape,
        subtomo_shape,
        subtomos_internal_path,
        class_number,
        overlap_thickness)
