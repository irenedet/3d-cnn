import pandas as pd
import numpy as np

from distutils.util import strtobool
from shutil import copyfile
from os.path import basename, join

from src.python.coordinates_toolbox.h5_subtomos import \
    compute_list_best_cross_correlation_angles
from src.python.performance.math_utils import radians2degrees

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-path_to_motl", "--path_to_motl",
                    help="",
                    type=str)
parser.add_argument("-catalogue_path", "--catalogue_path",
                    help="",
                    type=str)
parser.add_argument("-path_to_dataset", "--path_to_dataset",
                    help="",
                    type=str)
parser.add_argument("-ref_angles", "--reference_rotation_angles_file",
                    help="name of segmentation",
                    type=str)
parser.add_argument("-path_to_output_csv", "--path_to_output_csv",
                    help="",
                    type=str)
parser.add_argument("-tomo_name", "--tomo_name",
                    help="",
                    type=str)
parser.add_argument("-angles_in_degrees", "--angles_in_degrees",
                    help="",
                    type=str)
parser.add_argument("-path_to_mask", "--path_to_mask",
                    help="",
                    type=str)

args = parser.parse_args()
path_to_motl = args.path_to_motl
catalogue_path = args.catalogue_path
path_to_dataset = args.path_to_dataset
reference_rotation_angles_file = args.reference_rotation_angles_file
path_to_output_csv = args.path_to_output_csv
tomo_name = args.tomo_name
angles_in_degrees = args.angles_in_degrees
path_to_mask = args.path_to_mask

angles_in_degrees = strtobool(angles_in_degrees)
catalogue_basename = basename(catalogue_path)
tmp_catalogue_path = join("/tmp", tomo_name + catalogue_basename)

motive_list_df = pd.read_csv(path_to_motl, header=None)
list_of_peak_coordinates = list(np.array(motive_list_df.iloc[:, 7:10]))

copyfile(src=catalogue_path, dst=tmp_catalogue_path)
list_best_cross_correlations, list_of_angles = \
    compute_list_best_cross_correlation_angles(
        list_of_peak_coordinates=list_of_peak_coordinates,
        catalogue_path=tmp_catalogue_path,
        path_to_mask=path_to_mask,
        path_to_dataset=path_to_dataset,
        reference_rotation_angles_file=reference_rotation_angles_file,
        in_tom_format=True)

if not angles_in_degrees:
    orientation_angles_array = radians2degrees(list_of_angles)
else:
    orientation_angles_array = np.array(list_of_angles)

motive_list_df.iloc[:, 16:19] = orientation_angles_array

print("Adding cross correlation to the 11th column of the motive list.")
motive_list_df.iloc[:, 10] = list_best_cross_correlations

motive_list_df.to_csv(path_or_buf=path_to_output_csv, header=False,
                      index=False)

print("Motive list saved as", path_to_output_csv)
