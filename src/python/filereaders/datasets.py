import os
import numpy as np
from src.python.filereaders.em import read_em
from src.python.filereaders.hdf import _load_hdf_dataset


def load_dataset(path_to_dataset: str) -> np.array:
    """
    Verified that they open according to same coordinate system
    """
    global dataset
    _, data_file_extension = os.path.splitext(path_to_dataset)
    assert data_file_extension in [".em", ".hdf"], "file in non valid format."
    if data_file_extension == ".em":
        em_header, dataset = read_em(path_to_emfile=path_to_dataset)
    elif data_file_extension == ".hdf":
        dataset = _load_hdf_dataset(hdf_file_path=path_to_dataset)
    return dataset
