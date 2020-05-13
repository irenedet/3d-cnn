import mrcfile
import numpy as np


def write_mrc_dataset(mrc_path: str, array: np.array, dtype=np.float32):
    array = np.array(array, dtype=dtype)
    with mrcfile.new(mrc_path, overwrite=True) as mrc:
        mrc.set_data(array)
    print("Dataset saved in", mrc_path)
    return
