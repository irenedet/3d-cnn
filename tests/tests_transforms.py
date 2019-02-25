from src.python.datasets.transformations import ElasticTransform, RandomRot3D, \
    RandomFlip3D, AdditiveGaussianNoise
# import h5py
from os.path import join
import numpy as np
from src.python.filereaders import h5
# from src.python.image.viewers import view_slice

from src.python.naming import h5_internal_paths


def transform_data_from_h5(training_data_path: str, label_name: str,
                           number_iter: int, output_data_path: str, split: int):
    raw_data, labeled_data = h5.read_training_data(training_data_path,
                                                   label_name=label_name,
                                                   split=split)
    numb_train = raw_data.shape[0]
    print(numb_train)
    raw_data = raw_data[None, :]
    labeled_data = labeled_data[None, :]
    for iteration in range(number_iter):
        if iteration == 0:
            transform = False
        else:
            transform = True

        transformed_raw, transformed_labeled = transform_data(raw_data,
                                                              labeled_data,
                                                              transform)

        with h5py.File(output_data_path, 'a') as f:
            for img_index in range(numb_train):
                subtomo_name = str(iteration) + "_" + str(img_index)
                subtomo_raw_h5_path = h5_internal_paths.RAW_SUBTOMOGRAMS
                subtomo_raw_h5_path = join(subtomo_raw_h5_path, subtomo_name)

                subtomo_label_h5_path = h5_internal_paths.LABELED_SUBTOMOGRAMS
                subtomo_label_h5_path = join(subtomo_label_h5_path,
                                             label_name)
                subtomo_label_h5_path = join(subtomo_label_h5_path,
                                             subtomo_name)

                f[subtomo_raw_h5_path] = transformed_raw[0, img_index, :, :, :]
                f[subtomo_label_h5_path] = transformed_labeled[0, img_index, :,
                                           :, :]
    return


def transform_data(raw_data: np.array, labeled_data: np.array,
                   transform=True) -> tuple:
    if transform:
        sigma = 3 * np.random.random()
        alpha = 3 * np.random.random()
        transform = ElasticTransform(alpha=alpha, sigma=sigma)
        transformed_raw = transform(raw_data)
        transformed_labeled = transform(labeled_data)

        transform = RandomRot3D(rot_range=90,
                                p=0.2)  # erases the channel dimension
        transformed_raw = transform(transformed_raw)
        transformed_labeled = transform(transformed_labeled)

        transformed_raw = transformed_raw[None, :]
        transformed_labeled = transformed_labeled[None, :]

        # No flip, to avoid wrong particle chirality
        # transform = RandomFlip3D()  # erases the channel dimension
        # transformed_raw = transform(transformed_raw)
        # transformed_labeled = transform(transformed_labeled)
        #
        # transformed_raw = transformed_raw[None, :]
        # transformed_labeled = transformed_labeled[None, :]

        sigma = np.random.random()
        transform = AdditiveGaussianNoise(sigma=sigma)
        transformed_raw = transform(transformed_raw)
        transformed_labeled = transform(transformed_labeled)
        return transformed_raw, transformed_labeled
    else:
        print("The data is not being transformed")
        return raw_data, labeled_data


import h5py
from src.python.naming import h5_internal_paths


def write_transformed_data(training_data_path: str, transformed_raw: np.array,
                           transformed_labeled: np.array, label_name: str):
    with h5py.File(training_data_path, 'r') as f:
        raw_subtomo_names = list(f[h5_internal_paths.RAW_SUBTOMOGRAMS])
        for subtomo_name in raw_subtomo_names:
            print("ToDo")
    pass


training_data_path = \
    '/scratch/trueba/3d-cnn/training_data/training_data_side128_49examples.h5'

label_name = "ribosomes"
print("The training data path is ", training_data_path)
number_iter = 2
output_data_path = "/scratch/trueba/3d-cnn/training_data/data_aug_test.h5"

transform_data_from_h5(training_data_path=training_data_path,
                       label_name=label_name, number_iter=number_iter,
                       output_data_path=output_data_path, split=-1)
