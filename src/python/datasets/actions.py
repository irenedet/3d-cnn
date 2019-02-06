import numpy as np

from src.python.coordinates_toolbox.subtomos import \
    get_particle_coordinates_grid_with_overlap
from src.python.filewriters.h5 import write_subtomograms_from_dataset


def split_dataset(data: np.array, labels: np.array, split: int) -> tuple:
    train_data, train_labels = data[:split], labels[:split]
    val_data, val_labels = data[split:], labels[split:]
    print("Shape of training data:", train_data.shape)
    print("Shape of validation data", val_data.shape)
    return train_data, train_labels, val_data, val_labels


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


def partition_tomogram(dataset, output_file_path: str,
                       shape_to_crop_zyx=(128, 128, 128),
                       border_frame_thickness=12
                       ):
    padded_dataset = pad_dataset(dataset, shape_to_crop_zyx,
                                 border_frame_thickness)
    print("The padded dataset has shape", padded_dataset.shape)
    padded_particles_coordinates = get_particle_coordinates_grid_with_overlap(
        padded_dataset.shape,
        shape_to_crop_zyx,
        border_frame_thickness)
    write_subtomograms_from_dataset(output_file_path, padded_dataset,
                                    padded_particles_coordinates,
                                    shape_to_crop_zyx)
