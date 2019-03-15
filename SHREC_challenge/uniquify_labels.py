path_to_mask = "/struct/mahamid/Irene/shrec/0/all_classes_foreground_mask_0.hdf"
print(path_to_mask)
import h5py
from src.python.naming import h5_internal_paths
import numpy as np

with h5py.File(path_to_mask, "a") as f:
    print(list(f[""]))
    # data = f[h5_internal_paths.HDF_INTERNAL_PATH][:]
    # floor_data = np.floor(data)
    # f[h5_internal_paths.HDF_INTERNAL_PATH] = floor_data
