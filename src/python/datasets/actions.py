import numpy as np
import random

from src.python.coordinates_toolbox.subtomos import \
    get_particle_coordinates_grid_with_overlap
from src.python.filewriters.h5 import write_subtomograms_from_dataset, \
    write_joint_raw_and_labels_subtomograms


def split_dataset(data: np.array, labels: np.array, split: int) -> tuple:
    data = list(data)
    labels = list(labels)
    data_order = list(range(len(data)))
    combined = list(zip(data, labels, data_order))
    random.shuffle(combined)
    data[:], labels[:], data_order = zip(*combined)
    data = np.array(data)
    labels = np.array(labels)
    train_data, train_labels = data[:split], labels[:split]
    val_data, val_labels = data[split:], labels[split:]
    print("Shape of training data:", train_data.shape)
    print("Shape of validation data", val_data.shape)
    return train_data, train_labels, val_data, val_labels, data_order


def get_right_padding_lengths(tomo_shape, shape_to_crop_zyx):
    padding = [box_size - (tomo_size % box_size) for tomo_size, box_size
               in zip(tomo_shape, shape_to_crop_zyx)]
    return padding


def pad_dataset(dataset: np.array,
                cubes_with_border_shape: list,
                overlap_thickness: int):
    internal_cube_shape = [dim - 2 * overlap_thickness for dim in
                           cubes_with_border_shape]
    right_padding = get_right_padding_lengths(dataset.shape,
                                              internal_cube_shape)
    right_padding = [[overlap_thickness, padding + overlap_thickness] for
                     padding in right_padding]

    padded_dataset = np.pad(array=dataset, pad_width=right_padding,
                            mode="reflect")
    return padded_dataset


def partition_tomogram(dataset, output_h5_file_path: str,
                       subtomo_shape: tuple,
                       overlap: int
                       ):
    padded_dataset = pad_dataset(dataset, subtomo_shape,
                                 overlap)
    padded_particles_coordinates = get_particle_coordinates_grid_with_overlap(
        padded_dataset.shape,
        subtomo_shape,
        overlap)
    write_subtomograms_from_dataset(output_h5_file_path, padded_dataset,
                                    padded_particles_coordinates,
                                    subtomo_shape)


def partition_raw_and_labels_tomograms(raw_dataset: np.array,
                                       labels_dataset: np.array,
                                       label_name: str,
                                       output_h5_file_path: str,
                                       subtomo_shape: tuple,
                                       overlap: int
                                       ):
    padded_raw_dataset = pad_dataset(raw_dataset, subtomo_shape, overlap)
    padded_labels_dataset = pad_dataset(labels_dataset, subtomo_shape, overlap)
    padded_particles_coordinates = get_particle_coordinates_grid_with_overlap(
        padded_raw_dataset.shape,
        subtomo_shape,
        overlap)
    write_joint_raw_and_labels_subtomograms(
        output_path=output_h5_file_path,
        padded_raw_dataset=padded_raw_dataset,
        padded_labels_dataset=padded_labels_dataset,
        label_name=label_name,
        window_centers=padded_particles_coordinates,
        crop_shape=subtomo_shape)
