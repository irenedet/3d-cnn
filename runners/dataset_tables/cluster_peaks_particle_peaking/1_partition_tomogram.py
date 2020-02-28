from os import makedirs
import argparse
import numpy as np

from tomogram_utils.volume_actions.actions import partition_tomogram
from file_actions.readers.hdf import _load_hdf_dataset

parser = argparse.ArgumentParser()
parser.add_argument("-raw", "--path_to_raw",
                    help="path to tomogram to be segmented in hdf format",
                    type=str)
parser.add_argument("-output", "--output_dir",
                    help="directory where the outputs will be stored",
                    type=str)
parser.add_argument("-outh5", "--output_h5_path",
                    help="file where the outputs will be stored",
                    type=str)
parser.add_argument("-box", "--box_side",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-overlap", "--overlap",
                    help="name of category to be segmented",
                    type=int)

args = parser.parse_args()
path_to_raw = args.path_to_raw
output_dir = args.output_dir
output_h5_file_path = args.output_h5_path
box_side = args.box_side
overlap = args.overlap

subtomogram_shape = tuple(box_side * np.array([1, 1, 1]))
makedirs(name=output_dir, exist_ok=True)

raw_dataset = _load_hdf_dataset(hdf_file_path=path_to_raw)
partition_tomogram(dataset=raw_dataset, output_h5_file_path=output_h5_file_path,
                   subtomo_shape=subtomogram_shape,
                   overlap=overlap)
