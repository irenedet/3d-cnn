import argparse
import os
from distutils.util import strtobool
from os import makedirs
from os.path import join
from shutil import copyfile

import h5py
import numpy as np
import pandas as pd

from constants import h5_internal_paths
from networks.utils import data_loader
from tomogram_utils.volume_actions.random_transformations import RandomRot3D, \
    SinusoidalElasticTransform3D, AdditiveGaussianNoise, \
    AdditiveSaltAndPepperNoise


def get_transforms(rot_range: float, elastic_alpha: int,
                   sigma_noise: float, salt_pepper_p: float = 0.04,
                   salt_pepper_ampl: float = 0.8) -> tuple:
    """

    :param rot_range:
    :param elastic_alpha:
    :param sigma_noise:
    :param salt_pepper_p:
    :param salt_pepper_ampl:
    :return:
    """
    rotation_transform = RandomRot3D(rot_range=rot_range, p=0.5)

    gaussian_transform = AdditiveGaussianNoise(sigma=sigma_noise)
    salt_pepper_noise = AdditiveSaltAndPepperNoise(p=salt_pepper_p,
                                                   amplitude=salt_pepper_ampl)
    if elastic_alpha >= 1:

        elastic_transform = SinusoidalElasticTransform3D(alpha=elastic_alpha,
                                                         interp_step=32)
        raw_transforms = [rotation_transform, elastic_transform,
                          gaussian_transform, salt_pepper_noise]
        label_transforms = [rotation_transform, elastic_transform]

    else:
        raw_transforms = [rotation_transform, gaussian_transform,
                          salt_pepper_noise]
        label_transforms = [rotation_transform]

    # raw_transforms = [elastic_transform, gaussian_transform,
    # salt_pepper_noise]
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
                       elastic_alpha: int, sigma_noise: float,
                       salt_pepper_p: float, salt_pepper_ampl: float):
    raw_volume_transforms, label_volume_transforms = [], []

    for _ in range(volumes_number):
        raw_transforms, label_transforms = \
            get_transforms(rot_range=rot_range,
                           elastic_alpha=elastic_alpha,
                           sigma_noise=sigma_noise,
                           salt_pepper_p=salt_pepper_p,
                           salt_pepper_ampl=salt_pepper_ampl)

        raw_volume_transforms += [raw_transforms]
        label_volume_transforms += [label_transforms]

    return raw_volume_transforms, label_volume_transforms


def write_raw_tensor(dst_data, raw_tensor, iteration):
    print("data shape", raw_tensor.shape)
    volumes_number = raw_tensor.shape[0]
    print("Iteration", iteration, "for raw data")
    global_subtomo_raw_h5_path = h5_internal_paths.RAW_SUBTOMOGRAMS
    with h5py.File(dst_data, 'a') as f:
        for batch_id in range(volumes_number):
            subtomo_name = str(iteration) + "_" + str(batch_id)
            subtomo_raw_h5_path = join(global_subtomo_raw_h5_path, subtomo_name)
            f[subtomo_raw_h5_path] = raw_tensor[batch_id, 0, :, :, :]
    return


def write_label_tensor(dst_data, label_tensor, iteration, label_name):
    volumes_number = label_tensor.shape[0]
    print("Iteration", iteration, "for label", label_name)
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
    src_label_data = list()
    assert len(semantic_classes) > 0
    for label_name in semantic_classes:
        src_raw, src_label = data_loader(data_path=src_data_path,
                                         semantic_class=label_name,
                                         number_vols=number_vols,
                                         labeled_only=False)
        src_label_data += [src_label]

    src_raw_keep = list()
    src_label_data_keep = list()
    n = len(src_raw)
    print("Number of raw sub-tomograms", n)
    for index in range(n):
        label_maxima_list = [np.max(label[index]) for label in src_label_data]
        if np.max(label_maxima_list) > 0:
            label_data = [label[index] for label in src_label_data]
            src_raw_keep.append(src_raw[index])
            src_label_data_keep.append(label_data)

    print("Number of labeled sub-tomograms", len(src_raw_keep))
    src_raw_keep = np.array(src_raw_keep)
    if len(src_label_data_keep) > 0:
        src_label_data_keep = np.swapaxes(np.array(src_label_data_keep), 0, 1)
    src_label_data_keep = list(src_label_data_keep)
    return src_raw_keep, src_label_data_keep


def write_raw_and_labels_vols(dst_data_path, src_raw, src_label_data,
                              semantic_classes, iteration):
    write_raw_tensor(dst_data_path, src_raw, iteration)
    for label_name, src_label in zip(semantic_classes, src_label_data):
        write_label_tensor(dst_data_path, src_label, iteration, label_name)
    return


def apply_transformation_iteration(src_raw, src_label_data, rot_range,
                                   elastic_alpha, sigma_noise,
                                   salt_pepper_p, salt_pepper_ampl):
    volumes_number = src_raw.shape[0]
    raw_volume_transforms, label_volume_transforms = \
        get_transform_list(volumes_number=volumes_number,
                           rot_range=rot_range,
                           elastic_alpha=elastic_alpha,
                           sigma_noise=sigma_noise,
                           salt_pepper_p=salt_pepper_p,
                           salt_pepper_ampl=salt_pepper_ampl)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-segmentation_names", "--segmentation_names",
                        help="segmentation_names",
                        type=str)
    parser.add_argument("-dst_data_path", "--dst_data_path",
                        help="Destination file path",
                        type=str)
    parser.add_argument("-write_on_table", "--write_on_table",
                        help="if True, name of training set "
                             "will be recorded in db",
                        type=str)
    parser.add_argument("-data_aug_rounds", "--data_aug_rounds",
                        type=int)
    parser.add_argument("-rot_angle", "--rot_angle",
                        type=float)
    parser.add_argument("-elastic_alpha", "--elastic_alpha",
                        type=float)
    parser.add_argument("-sigma_noise", "--sigma_noise",
                        type=float)
    parser.add_argument("-salt_pepper_p", "--salt_pepper_p",
                        type=float, default=0.04)
    parser.add_argument("-salt_pepper_ampl", "--salt_pepper_ampl",
                        type=float, default=0.8)
    parser.add_argument("-epsilon", "--epsilon",
                        type=float)
    parser.add_argument("-src_data_path", "--src_data_path",
                        help="path to src_data_path in .h5 format",
                        type=str)
    parser.add_argument("-dataset_table", "--dataset_table",
                        help="path to db (dataset_table) in .csv format",
                        type=str)
    parser.add_argument("-tomo_name", "--tomo_name",
                        help="tomo_name in sessiondate/datanumber format",
                        type=str)
    parser.add_argument("-output_column", "--output_column",
                        help="name of output_column in dataset table where "
                             "the partition path will be recorded",
                        type=str)

    args = parser.parse_args()
    tomo_name = args.tomo_name
    dataset_table = args.dataset_table
    dst_data_path = args.dst_data_path
    segmentation_names = args.segmentation_names
    data_aug_rounds = args.data_aug_rounds
    rot_angle = args.rot_angle
    sigma_noise = args.sigma_noise
    # epsilon = args.epsilon
    salt_pepper_p = args.salt_pepper_p
    salt_pepper_ampl = args.salt_pepper_ampl
    elastic_alpha = args.elastic_alpha
    src_data_path = args.src_data_path
    write_on_table = strtobool(args.write_on_table)
    folder_name = segmentation_names + "_DA"
    output_column = args.output_column

    output_dir = os.path.dirname(dst_data_path)
    makedirs(output_dir, exist_ok=True)

    semantic_classes = segmentation_names.split(',')
    print("semantic_classes", semantic_classes)

    if os.path.exists(dst_data_path):
        print("Data-augmented partition already exists.")
    else:
        src_raw, src_label_data = \
            get_raw_and_labels_vols(src_data_path=src_data_path,
                                    semantic_classes=semantic_classes)
        # print("src_raw.shape", src_raw.shape)
        if src_raw.shape[0] > 0:
            # Copying the original data
            iteration = -1
            write_raw_and_labels_vols(dst_data_path=dst_data_path,
                                      src_raw=src_raw,
                                      src_label_data=src_label_data,
                                      semantic_classes=semantic_classes,
                                      iteration=iteration)

            # Starting data augmentation:
            for iteration in range(data_aug_rounds):
                transf_raw_tensor, transf_label_tensors = \
                    apply_transformation_iteration(
                        src_raw, src_label_data, rot_range=rot_angle,
                        elastic_alpha=elastic_alpha,
                        sigma_noise=sigma_noise,
                        salt_pepper_p=0,
                        salt_pepper_ampl=0)

                write_raw_and_labels_vols(dst_data_path=dst_data_path,
                                          src_raw=transf_raw_tensor,
                                          src_label_data=transf_label_tensors,
                                          semantic_classes=semantic_classes,
                                          iteration=iteration)
        else:
            print("partition was empty, so is DA")

            copyfile(src_data_path, dst_data_path)
    if write_on_table:
        dataset_table = args.dataset_table
        tomo_name = args.tomo_name
        print("Writing path and DA data on table:", dataset_table)
        print("Training partition written on table: ", dst_data_path)
        df = pd.read_csv(dataset_table)
        df['tomo_name'] = df['tomo_name'].astype(str)
        tomo_df = df[df['tomo_name'] == tomo_name]
        df.loc[df['tomo_name'] == tomo_name, output_column] = dst_data_path
        # df.loc[df['tomo_name'] == tomo_name, 'rot_angle'] = rot_angle
        # df.loc[df['tomo_name'] == tomo_name, 'elastic_alpha'] = elastic_alpha
        # df.loc[df['tomo_name'] == tomo_name, 'sigma_noise'] = sigma_noise
        # df.loc[df['tomo_name'] == tomo_name, 'salt_pepper_p'] = salt_pepper_p
        # df.loc[
        #     df['tomo_name'] == tomo_name, 'data_aug_rounds'] = data_aug_rounds
        df.to_csv(path_or_buf=dataset_table, index=False)
