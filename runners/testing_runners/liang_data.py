import h5py
import numpy as np
from src.python.naming import h5_internal_paths

lamella_path = "/struct/mahamid/Irene/liang_data/lamella.hdf"
shape = (928, 928, 450)
with h5py.File(lamella_path, 'w') as f:
    f[h5_internal_paths.HDF_INTERNAL_PATH] = np.ones(shape)
