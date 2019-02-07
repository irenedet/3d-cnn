from os.path import join

import h5py

from src.python.datasets.actions import partition_tomogram
from src.python.naming import h5_internal_paths

path_to_raw = '/scratch/trueba/cnn/004/4bin/cnn/rawtomogram/001_bin4_subregion0-0-380-927-927-600.hdf'
with h5py.File(path_to_raw, 'r') as f:
    raw_dataset = f[h5_internal_paths.HDF_INTERNAL_PATH][:]

folder_path = "/scratch/trueba/3d-cnn/TEST/"
h5_output_file = "004_in_subtomos_128side_with_overlap.h5"
output_file_path = join(folder_path, h5_output_file)
partition_tomogram(dataset=raw_dataset, output_h5_file_path=output_file_path,
                   subtomo_shape=(128, 128, 128))
