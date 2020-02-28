import os

import numpy as np

from file_actions.readers.csv import read_motl_from_csv
from file_actions.readers.em import read_em


def load_motl(path_to_dataset: str) -> np.array:
    _, data_file_extension = os.path.splitext(path_to_dataset)
    assert data_file_extension in [".em", ".csv"], "file in non valid format."
    if data_file_extension == ".em":
        em_header, motl = read_em(path_to_emfile=path_to_dataset)
    elif data_file_extension == ".csv":
        motl = read_motl_from_csv(path_to_csv_motl=path_to_dataset)
    return motl
