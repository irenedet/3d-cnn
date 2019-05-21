from os import makedirs
import argparse
import numpy as np

from src.python.datasets.actions import partition_raw_intersecting_mask
from src.python.filereaders.hdf import _load_hdf_dataset

parser = argparse.ArgumentParser()
parser.add_argument("-raw", "--path_to_raw",
                    help="path to tomogram to be segmented in hdf format",
                    type=str)
parser.add_argument("-hdf_lamella_file", "--hdf_lamella_file",
                    help="path to lamella mask in hdf format",
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
path_to_lamella = args.hdf_lamella_file
output_dir = args.output_dir
output_h5_file_path = args.output_h5_path
box_side = args.box_side
overlap = args.overlap

subtomogram_shape = tuple(box_side * np.array([1, 1, 1]))
makedirs(name=output_dir, exist_ok=True)

raw_dataset = _load_hdf_dataset(hdf_file_path=path_to_raw)
lamella_mask = _load_hdf_dataset(hdf_file_path=path_to_lamella)

lamella_shape = lamella_mask.shape
dataset_shape = raw_dataset.shape
print("lamella_shape", lamella_shape, "dataset_shape", dataset_shape)

minimum_shape = [np.min([data_dim, lamella_dim]) for data_dim, lamella_dim in
                 zip(dataset_shape, lamella_shape)]
minz, miny, minx = minimum_shape
print(minz, miny, minx)

lamella_mask = lamella_mask[:minz, :miny, :minx]
raw_dataset = raw_dataset[:minz, :miny, :minx]

partition_raw_intersecting_mask(dataset=raw_dataset,
                                mask_dataset=lamella_mask,
                                output_h5_file_path=output_h5_file_path,
                                subtomo_shape=subtomogram_shape,
                                overlap=overlap)
