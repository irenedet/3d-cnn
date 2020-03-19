import argparse
from distutils.util import strtobool
from os import makedirs

import numpy as np
import pandas as pd

from file_actions.readers.tomograms import load_tomogram
from tomogram_utils.volume_actions.actions import \
    partition_raw_intersecting_mask
from tomogram_utils.volume_actions.actions import partition_tomogram

parser = argparse.ArgumentParser()
parser.add_argument("-dataset_table", "--dataset_table",
                    help="path to dataset table",
                    type=str)
parser.add_argument("-tomo_name", "--tomo_name",
                    help="string, name of tomogram to partition",
                    type=str)
parser.add_argument("-output", "--output_dir",
                    help="directory where the outputs will be stored",
                    type=str)
parser.add_argument("-outh5", "--output_h5_path",
                    help="file where the outputs will be stored",
                    type=str)
parser.add_argument("-box", "--box_side",
                    help="name of category to be segmented",
                    type=str)
parser.add_argument("-overlap", "--overlap",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-write_on_table", "--write_on_table",
                    help="Boolean to choose if writing on table",
                    type=str, default='False')
parser.add_argument("-rel_bin_lamella_mask", "--rel_bin_lamella_mask",
                    help="A bin of 1 means that lamella is one time binned",
                    type=int, default=0)

args = parser.parse_args()
dataset_table = args.dataset_table
tomo_name = args.tomo_name
output_dir = args.output_dir
output_h5_file_path = args.output_h5_path
box_shape = args.box_side
overlap = args.overlap
rel_bin_lamella_mask = args.rel_bin_lamella_mask
write_on_table = strtobool(args.write_on_table)
box_shape = box_shape.split(',')

assert len(box_shape) in [1, 3], "Invalid box_side"
if len(box_shape) == 1:
    subtomogram_shape = tuple(box_shape * np.array([1, 1, 1]))
else:
    subtomogram_shape = tuple([int(elem) for elem in reversed(box_shape)])

makedirs(output_dir, exist_ok=True)

if rel_bin_lamella_mask == 0:
    print("Lamella mask and original raw files have the same binning.")
elif rel_bin_lamella_mask > 0:
    print("Lamella mask is ", rel_bin_lamella_mask,
          "binned w.r.t original raw files.")
else:
    print("this should never happen!")

df = pd.read_csv(dataset_table)
tomo_df = df[df['tomo_name'] == tomo_name]
path_to_raw = tomo_df.iloc[0]['eman2_filetered_tomo']
path_to_lamella = tomo_df.iloc[0]['lamella_file']

raw_dataset = load_tomogram(path_to_dataset=path_to_raw)
print("raw_dataset.shape", raw_dataset.shape)
if isinstance(path_to_lamella, float):
    print("No lamella file available.")
    partition_tomogram(dataset=raw_dataset,
                       output_h5_file_path=output_h5_file_path,
                       subtomo_shape=subtomogram_shape, overlap=overlap)
else:
    path_to_lamella = tomo_df.iloc[0]['lamella_file']
    lamella_mask = load_tomogram(path_to_dataset=path_to_lamella)

    lamella_shape = lamella_mask.shape
    dataset_shape = raw_dataset.shape
    print("lamella_shape", lamella_shape, "dataset_shape", dataset_shape)

    minimum_shape = [np.min([data_dim, lamella_dim]) for data_dim, lamella_dim
                     in zip(dataset_shape, lamella_shape)]
    minz, miny, minx = minimum_shape

    lamella_mask = lamella_mask[:minz, :miny, :minx]
    raw_dataset = raw_dataset[:minz, :miny, :minx]

    partition_raw_intersecting_mask(dataset=raw_dataset,
                                    mask_dataset=lamella_mask,
                                    output_h5_file_path=output_h5_file_path,
                                    subtomo_shape=subtomogram_shape,
                                    overlap=overlap)

if write_on_table:
    df.loc[df['tomo_name'] == tomo_name, 'test_partition'] = output_h5_file_path
    df.to_csv(dataset_table, index=False)
