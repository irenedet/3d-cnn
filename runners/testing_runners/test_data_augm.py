import h5py
import numpy as np
import os
from os.path import join

from src.python.naming import h5_internal_paths
from src.python.networks.utils import data_loader
from src.python.datasets.random_transformations import RandomRot3D, \
    SinusoidalElasticTransform3D, AdditiveGaussianNoise


def get_transforms(rot_range: float, elastic_alpha: int,
                   sigma_noise: float) -> tuple:
    rotation_transform = RandomRot3D(rot_range=rot_range, p=1)
    elastic_transform = SinusoidalElasticTransform3D(alpha=elastic_alpha,
                                                     interp_step=32)
    gaussian_transform = AdditiveGaussianNoise(sigma=sigma_noise)

    # Todo: check if rotation transforms are ok: they dont transform the same way all the volumes...
    raw_transforms = [rotation_transform, elastic_transform, gaussian_transform]
    label_transforms = [rotation_transform, elastic_transform]

    # raw_transforms = [rotation_transform, gaussian_transform]
    # label_transforms = [rotation_transform]

    # raw_transforms = [elastic_transform, gaussian_transform]
    # label_transforms = [elastic_transform]

    return raw_transforms, label_transforms


def apply_transforms_to_batch(tensor, volume_transforms):
    assert len(volume_transforms) == tensor.shape[0]
    transformed_tensor = []
    for batch_id, transforms in enumerate(volume_transforms):
        volume = tensor[batch_id, 0, :, :, :]
        for transform in transforms:
            volume = transform._apply_volume_function(tensor=volume)

        transformed_tensor += [volume]

    transformed_tensor = np.array(transformed_tensor)
    transformed_tensor = transformed_tensor[:, None]
    return transformed_tensor


def get_transform_list(volumes_number: int, rot_range: float,
                       elastic_alpha: int, sigma_noise: float):
    raw_volume_transforms, label_volume_transforms = [], []

    for _ in range(volumes_number):
        raw_transforms, label_transforms = \
            get_transforms(rot_range=rot_range,
                           elastic_alpha=elastic_alpha,
                           sigma_noise=sigma_noise)

        raw_volume_transforms += [raw_transforms]
        label_volume_transforms += [label_transforms]

    return raw_volume_transforms, label_volume_transforms


def write_raw_tensor(dst_data, raw_tensor, iteration):
    print("raw_tensor.shape", raw_tensor.shape)
    volumes_number = raw_tensor.shape[0]
    with h5py.File(dst_data, 'a') as f:
        for batch_id in range(volumes_number):
            subtomo_name = str(iteration) + "_" + str(batch_id)
            subtomo_raw_h5_path = h5_internal_paths.RAW_SUBTOMOGRAMS

            subtomo_raw_h5_path = join(subtomo_raw_h5_path, subtomo_name)
            f[subtomo_raw_h5_path] = raw_tensor[batch_id, 0, :, :, :]
    return


def write_label_tensor(dst_data, label_tensor, iteration, label_name):
    volumes_number = label_tensor.shape[0]
    with h5py.File(dst_data, 'a') as f:
        for batch_id in range(volumes_number):
            subtomo_name = str(iteration) + "_" + str(batch_id)
            subtomo_label_h5_path = h5_internal_paths.LABELED_SUBTOMOGRAMS
            subtomo_label_h5_name = join(subtomo_label_h5_path, label_name)
            subtomo_label_h5_name = join(subtomo_label_h5_name, subtomo_name)
            f[subtomo_label_h5_name] = label_tensor[batch_id, 0, :, :, :]
    return


def get_raw_and_labels_vols(src_data_path, semantic_classes,
                            number_vols: int = -1):
    src_label_data = []
    for label_name in semantic_classes:
        src_raw, src_label = data_loader(data_path=src_data_path,
                                         semantic_class=label_name,
                                         number_vols=number_vols)

        src_label_data += [src_label]
    return src_raw, src_label_data


def write_raw_and_labels_vols(dst_data_path, src_raw, src_label_data,
                              semantic_classes, iteration):
    write_raw_tensor(dst_data_path, src_raw, iteration)
    for label_name, src_label in zip(semantic_classes, src_label_data):
        write_label_tensor(dst_data_path, src_label, iteration, label_name)
    return


def apply_transformation_iteration(src_raw, src_label_data, rot_range,
                                   elastic_alpha, sigma_noise):
    volumes_number = src_raw.shape[0]
    raw_volume_transforms, label_volume_transforms = \
        get_transform_list(volumes_number=volumes_number,
                           rot_range=rot_range,
                           elastic_alpha=elastic_alpha,
                           sigma_noise=sigma_noise)
    # print("len(raw_volume_transforms)", len(raw_volume_transforms))
    transf_raw_tensor = \
        apply_transforms_to_batch(tensor=src_raw,
                                  volume_transforms=raw_volume_transforms)

    transf_label_tensors = []
    for src_label in src_label_data:
        transf_label_tensor = \
            apply_transforms_to_batch(tensor=src_label,
                                      volume_transforms=label_volume_transforms)
        transf_label_tensors += [transf_label_tensor]

    return transf_raw_tensor, transf_label_tensors


# src_data_path = "/home/papalotl/test_data_aug.h5"
# dst_data_path = "/home/papalotl/DA_new_test.h5"
# semantic_classes = ['ribo']

src_data_path = "/struct/mahamid/Irene/yeast/healthy/180426/004/G_sigma1_non_sph/train_and_test_partitions/full_partition.h5"
dst_data_path = "/struct/mahamid/Irene/yeast/healthy/180426/004/G_sigma1_non_sph/train_and_test_partitions/DA_gauss_elast_rot.h5"
# src_data_path = "/struct/mahamid/Irene/yeast/healthy/180426/004/G_sigma1_non_sph/train_and_test_partitions/dummy_test.h5"
# dst_data_path = "/struct/mahamid/Irene/yeast/healthy/180426/004/G_sigma1_non_sph/train_and_test_partitions/dummy_gauss_elast_rot.h5"

if os.path.exists(dst_data_path):
    os.remove(dst_data_path)

semantic_classes = ['ribo', 'fas', 'memb']

src_raw, src_label_data = \
    get_raw_and_labels_vols(src_data_path=src_data_path,
                            semantic_classes=semantic_classes)
print("src_raw.shape", src_raw.shape)
# Copying the original data
iteration = -1
write_raw_and_labels_vols(dst_data_path=dst_data_path, src_raw=src_raw,
                          src_label_data=src_label_data,
                          semantic_classes=semantic_classes,
                          iteration=iteration)

# Starting data augmentation:
data_aug_rounds = 2
rot_range = 60
elastic_alpha = 2
sigma_noise = 1.5
# volumes_number = 3  # src_raw.shape[0]

for iteration in range(data_aug_rounds):
    transf_raw_tensor, transf_label_tensors = apply_transformation_iteration(
        src_raw, src_label_data, rot_range=rot_range,
        elastic_alpha=elastic_alpha,
        sigma_noise=sigma_noise)

    write_raw_and_labels_vols(dst_data_path=dst_data_path,
                              src_raw=transf_raw_tensor,
                              src_label_data=transf_label_tensors,
                              semantic_classes=semantic_classes,
                              iteration=iteration)
