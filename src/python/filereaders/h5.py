from os.path import join

import h5py
import numpy as np

from src.python.naming import h5_internal_paths


def read_training_data(training_data_path: str,
                       label_name="ribosomes",
                       split=-1) -> tuple:
    data = []
    labels = []
    if split < 0:
        print("split = ", split)
        with h5py.File(training_data_path, 'r') as f:
            raw_subtomo_names = list(f[h5_internal_paths.RAW_SUBTOMOGRAMS])
            for subtomo_name in raw_subtomo_names:
                raw_subtomo_h5_internal_path = join(
                    h5_internal_paths.RAW_SUBTOMOGRAMS, subtomo_name)
                data += [f[raw_subtomo_h5_internal_path][:]]
                labels_subtomo_h5_internal_path = join(
                    h5_internal_paths.LABELED_SUBTOMOGRAMS, label_name)
                labels_subtomo_h5_internal_path = join(
                    labels_subtomo_h5_internal_path,
                    subtomo_name)
                labels += [f[labels_subtomo_h5_internal_path][:]]
    else:
        with h5py.File(training_data_path, 'r') as f:
            raw_subtomo_names = list(f[h5_internal_paths.RAW_SUBTOMOGRAMS])
            if 1 > split > 0:
                split = int(split*len(raw_subtomo_names))
            else:
                split = int(split)
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

def read_training_data_dice_multi_class(training_data_path: str,
                       segmentation_names:list,
                       split=-1) -> tuple:
    data = []
    labels = []
    with h5py.File(training_data_path, 'r') as f:
        raw_subtomo_names = list(f[h5_internal_paths.RAW_SUBTOMOGRAMS])
        for subtomo_name in raw_subtomo_names[:split]:
            raw_subtomo_h5_internal_path = join(
                h5_internal_paths.RAW_SUBTOMOGRAMS, subtomo_name)
            data += [f[raw_subtomo_h5_internal_path][:]]
            labels_current_subtomo = []
            for label_name in segmentation_names:
                labels_subtomo_h5_internal_path = join(
                    h5_internal_paths.LABELED_SUBTOMOGRAMS, label_name)
                labels_subtomo_h5_internal_path = join(
                    labels_subtomo_h5_internal_path,
                    subtomo_name)
                labels_current_subtomo += [f[labels_subtomo_h5_internal_path][:]]
            labels += [np.array(labels_current_subtomo)]

    data = np.array(data)
    labels = np.array(labels)
    print("Loaded data of shape", data.shape)
    print("Loaded labels of shape", labels.shape)
    return data, labels

def read_raw_data_from_h5(data_path: str) -> np.array:
    data = []
    with h5py.File(data_path, 'r') as f:
        for subtomo_name in list(f[h5_internal_paths.RAW_SUBTOMOGRAMS]):
            subtomo_h5_internal_path = join(h5_internal_paths.RAW_SUBTOMOGRAMS,
                                            subtomo_name)
            data += [f[subtomo_h5_internal_path][:]]
    return np.array(data)