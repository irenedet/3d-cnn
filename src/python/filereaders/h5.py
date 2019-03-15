from os.path import join

import h5py
import numpy as np

from src.python.naming import h5_internal_paths


def read_training_data(training_data_path: str,
                       label_name="ribosomes",
                       split=-1) -> tuple:
    data = []
    labels = []
    with h5py.File(training_data_path, 'r') as f:
        raw_subtomo_names = list(f[h5_internal_paths.RAW_SUBTOMOGRAMS])
        for subtomo_name in raw_subtomo_names[:split]:
            raw_subtomo_h5_internal_path = join(
                h5_internal_paths.RAW_SUBTOMOGRAMS, subtomo_name)
            data += [f[raw_subtomo_h5_internal_path][:]]
            labels_subtomo_h5_internal_path = join(
                h5_internal_paths.LABELED_SUBTOMOGRAMS, label_name)
            labels_subtomo_h5_internal_path = join(
                labels_subtomo_h5_internal_path,
                subtomo_name)
            labels += [f[labels_subtomo_h5_internal_path][:]]

    data = np.array(data)
    labels = np.array(labels)
    assert data.shape == labels.shape
    print("Loaded data and labels of shape", labels.shape)

    return data, labels


def read_raw_data_from_h5(data_path: str) -> np.array:
    data = []
    with h5py.File(data_path, 'r') as f:
        for subtomo_name in list(f[h5_internal_paths.RAW_SUBTOMOGRAMS]):
            subtomo_h5_internal_path = join(h5_internal_paths.RAW_SUBTOMOGRAMS,
                                            subtomo_name)
            data += [f[subtomo_h5_internal_path][:]]
    return np.array(data)