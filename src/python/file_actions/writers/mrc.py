import mrcfile
import numpy as np


def write_mrc_dataset(mrc_path: str, array: np.array):
    array = np.array(array, dtype=np.float32)
    with mrcfile.new(mrc_path, overwrite=True) as mrc:
        mrc.set_data(array)
    return
