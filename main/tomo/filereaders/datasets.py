import os
import numpy as np
from main.tomo.filereaders.em import read_em
from main.tomo.filereaders.h5 import read_hdf


def load_dataset(path_to_dataset: str) -> np.array:
    """
    Verified that they open according to same coordinate system
    """
    global dataset
    _, data_file_extension = os.path.splitext(path_to_dataset)
    assert data_file_extension in [".em", ".hdf",
                                   ".csv"], "file in non valid format."
    if data_file_extension == ".em":
        em_header, dataset = read_em(path_to_emfile=path_to_dataset)
    elif data_file_extension == ".hdf":
        dataset = read_hdf(hdf_file_path=path_to_dataset)
    return dataset
