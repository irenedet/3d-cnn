import numpy as np
import random

from src.python.filereaders.hdf import _load_hdf_dataset
from src.python.coordinates_toolbox.subtomos import \
    get_particle_coordinates_grid_with_overlap
from src.python.filewriters.h5 import write_subtomograms_from_dataset, \
    write_joint_raw_and_labels_subtomograms, \
    write_joint_raw_and_labels_subtomograms_dice_multiclass


def split_dataset(data: np.array, labels: np.array, split: int,
                  shuffle=True) -> tuple:
    data = list(data)
    labels = list(labels)
    data_order = list(range(len(data)))
    combined = list(zip(data, labels, data_order))
    if shuffle:
        random.shuffle(combined)
    else:
        print("Splitting sets without shuffling")
    data[:], labels[:], data_order = zip(*combined)
    data = np.array(data)
    labels = np.array(labels)
    if isinstance(split, int):
        train_data, train_labels = data[:split], labels[:split]
        val_data, val_labels = data[split:], labels[split:]
    elif isinstance(split, tuple):
        assert len(split) == 2
        train_data0, train_labels0 = data[:split[0]], labels[:split[0]]
        train_data1, train_labels1 = data[split[1]:], labels[split[1]:]
        train_data = np.concatenate((train_data0, train_data1), axis=0)
        train_labels = np.concatenate((train_labels0, train_labels1), axis=0)

        val_data, val_labels = data[split[0]:split[1]], labels[
                                                        split[0]:split[1]]
    else:
        print("split should be an integer or a tuple of integers")
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


def partition_raw_and_labels_tomograms_dice_multiclass(
        path_to_raw: str,
        labels_dataset_list: list,
        segmentation_names: list,
        output_h5_file_path: str,
        subtomo_shape: tuple,
        overlap: int
):
    raw_dataset = _load_hdf_dataset(hdf_file_path=path_to_raw)
    padded_raw_dataset = pad_dataset(raw_dataset, subtomo_shape, overlap)
    padded_particles_coordinates = get_particle_coordinates_grid_with_overlap(
        padded_raw_dataset.shape,
        subtomo_shape,
        overlap)
    padded_labels_dataset_list = []
    for path_to_labeled in labels_dataset_list:
        labels_dataset = _load_hdf_dataset(hdf_file_path=path_to_labeled)
        padded_labels_dataset = pad_dataset(labels_dataset, subtomo_shape, overlap)
        padded_labels_dataset_list += [padded_labels_dataset]

    write_joint_raw_and_labels_subtomograms_dice_multiclass(
        output_path=output_h5_file_path,
        padded_raw_dataset=padded_raw_dataset,
        padded_labels_dataset_list=padded_labels_dataset_list,
        segmentation_names=segmentation_names,
        window_centers=padded_particles_coordinates,
        crop_shape=subtomo_shape)

# def partition_tomograms(dataset: np.array,
#                         label_name: str,
#                         output_h5_file_path: str,
#                         subtomo_shape: tuple,
#                         overlap: int
#                         ):
#     padded_raw_dataset = pad_dataset(dataset, subtomo_shape, overlap)
#     padded_particles_coordinates = get_particle_coordinates_grid_with_overlap(
#         padded_raw_dataset.shape,
#         subtomo_shape,
#         overlap)
#     write_subtomograms(
#         output_path=output_h5_file_path,
#         padded_raw_dataset=padded_raw_dataset,
#         padded_labels_dataset=padded_labels_dataset,
#         label_name=label_name,
#         window_centers=padded_particles_coordinates,
#         crop_shape=subtomo_shape)
#
#
# def write_subtomograms(output_path: str,
#                                             padded_raw_dataset: np.array,
#                                             padded_labels_dataset: np.array,
#                                             label_name: str,
#                                             window_centers: list,
#                                             crop_shape: tuple):
#     with h5py.File(output_path, 'w') as f:
#         for window_center in window_centers:
#             subtomo_name = "subtomo_{0}".format(str(window_center))
#             subtomo_raw_h5_internal_path = join(
#                 h5_internal_paths.RAW_SUBTOMOGRAMS,
#                 subtomo_name)
#             subtomo_raw_data = crop_window_around_point(
#                 input=padded_raw_dataset,
#                 crop_shape=crop_shape,
#                 window_center=window_center)
#
#             subtomo_label_h5_internal_path = join(
#                 h5_internal_paths.LABELED_SUBTOMOGRAMS, label_name)
#             subtomo_label_h5_internal_path = join(
#                 subtomo_label_h5_internal_path,
#                 subtomo_name)
#             subtomo_label_data = crop_window_around_point(
#                 input=padded_labels_dataset,
#                 crop_shape=crop_shape,
#                 window_center=window_center)
#             print("subtomo_max = ", np.max(subtomo_label_data))
#             if np.max(subtomo_label_data) > 0.2:
#                 f[subtomo_raw_h5_internal_path] = subtomo_raw_data
#                 f[subtomo_label_h5_internal_path] = subtomo_label_data
#             else:
#                 print("subtomo ", subtomo_name, "discarded")
#     return
