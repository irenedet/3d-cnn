from os.path import join

import h5py
import numpy as np


def read_training_data(training_data_path: str, cathegory="ribosomes") -> tuple:
    data = []
    labels = []
    with h5py.File(training_data_path, 'r') as f:
        raw_subtomo_names = list(f['volumes/raw'])
        for subtomo_name in raw_subtomo_names:
            raw_subtomo_h5_internal_path = join('volumes/raw', subtomo_name)
            data += [f[raw_subtomo_h5_internal_path][:]]
            labels_subtomo_h5_internal_path = join('volumes/labels/', cathegory)
            labels_subtomo_h5_internal_path = join(
                labels_subtomo_h5_internal_path,
                subtomo_name)
            labels += [f[labels_subtomo_h5_internal_path][:]]

    data = np.array(data)
    labels = np.array(labels)
    assert data.shape == labels.shape
    print("Loaded data and labels of shape", labels.shape)

    return data, labels

