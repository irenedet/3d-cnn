import h5py
from os.path import join
import os

import random
import torch
import numpy as np

from src.python.filereaders.csv import read_motl_from_csv
from src.python.filereaders.em import load_em_motl
from src.python.filereaders.shrec import read_shrec_motl
from src.python.coordinates_toolbox.utils import \
    extract_coordinates_from_em_motl, extract_coordinates_from_txt_shrec
from src.python.peak_toolbox.utils import paste_sphere_in_dataset

from src.python.naming import h5_internal_paths
from src.python.coordinates_toolbox import subtomos
from src.python.tensors.actions import crop_window_around_point
from src.python.pytorch_cnn.classes.unet import UNet
from src.python.filereaders.shrec import particle_dict


def write_dataset_hdf(output_path: str, tomo_data: np.array):
    with h5py.File(output_path, 'w') as f:
        f[h5_internal_paths.HDF_INTERNAL_PATH] = tomo_data
    print("The hdf file has been writen in ", output_path)


def write_dataset_from_subtomograms(output_path, subtomo_path, output_shape,
                                    subtomo_shape,
                                    subtomos_internal_path):
    tomo_data = np.zeros(output_shape)
    with h5py.File(subtomo_path, 'r') as f:
        for subtomo_name in list(f[subtomos_internal_path]):
            subtomo_center = subtomos.get_coord_from_name(subtomo_name)
            init_points, end_points, lengths = subtomos.get_subtomo_corners(
                output_shape,
                subtomo_shape,
                subtomo_center)
            print(init_points, end_points, lengths)
            subtomo_h5_internal_path = join(subtomos_internal_path,
                                            subtomo_name)
            tomo_data[init_points[0]: end_points[0],
            init_points[1]: end_points[1], init_points[2]: end_points[2]] = \
                f[subtomo_h5_internal_path][0, 0:lengths[0], 0:lengths[1],
                0:lengths[2]]
    write_dataset_hdf(output_path, tomo_data)
    print("right before deleting", np.max(tomo_data))
    del tomo_data


def write_dataset_from_subtomos_with_overlap(output_path,
                                             subtomo_path,
                                             output_shape,
                                             subtomo_shape,
                                             subtomos_internal_path,
                                             overlap):
    output_shape_with_overlap = output_shape  # [dim + overlap_thickness for
    # dim in
    # output_shape]
    print("The actual output shape is", output_shape_with_overlap)
    tomo_data = np.zeros(output_shape_with_overlap)

    internal_subtomo_shape = tuple([subtomo_dim - 2 * overlap for
                                    subtomo_dim in subtomo_shape])
    with h5py.File(subtomo_path, 'r') as f:
        for subtomo_name in list(f[subtomos_internal_path]):
            subtomo_center = subtomos.get_coord_from_name(subtomo_name)
            start_corner, end_corner, lengths = subtomos.get_subtomo_corners(
                output_shape,
                internal_subtomo_shape,
                subtomo_center)
            overlap_shift = overlap * np.array([1, 1, 1])
            start_corner -= overlap_shift
            end_corner -= overlap_shift
            subtomo_h5_internal_path = join(subtomos_internal_path,
                                            subtomo_name)

            internal_subtomo_data = f[subtomo_h5_internal_path][0,
                                    overlap:lengths[0] + overlap,
                                    overlap:lengths[1] + overlap,
                                    overlap:lengths[2] + overlap]

            tomo_data[start_corner[0]: end_corner[0],
            start_corner[1]: end_corner[1],
            start_corner[2]: end_corner[2]] = internal_subtomo_data
            print("internal_subtomo_data = ",
                  internal_subtomo_data.shape)
    write_dataset_hdf(output_path, tomo_data)
    print("right before deleting the maximum is", np.max(tomo_data))
    del tomo_data


def write_dataset_from_subtomos_with_overlap_multiclass(output_path,
                                                        subtomo_path,
                                                        output_shape,
                                                        subtomo_shape,
                                                        subtomos_internal_path,
                                                        class_number,
                                                        overlap):
    output_shape_with_overlap = output_shape  # [dim + overlap_thickness for
    # dim in
    # output_shape]
    print("The output shape is", output_shape_with_overlap)
    tomo_data = np.zeros(output_shape_with_overlap)

    internal_subtomo_shape = tuple([subtomo_dim - 2 * overlap for
                                    subtomo_dim in subtomo_shape])
    with h5py.File(subtomo_path, 'r') as f:
        for subtomo_name in list(f[subtomos_internal_path]):
            subtomo_center = subtomos.get_coord_from_name(subtomo_name)
            start_corner, end_corner, lengths = subtomos.get_subtomo_corners(
                output_shape,
                internal_subtomo_shape,
                subtomo_center)
            overlap_shift = overlap * np.array([1, 1, 1])
            start_corner -= overlap_shift
            end_corner -= overlap_shift
            subtomo_h5_internal_path = join(subtomos_internal_path,
                                            subtomo_name)
            channels = f[subtomo_h5_internal_path][:].shape[0]
            internal_subtomo_data = np.zeros(lengths)
            if channels > 1:
                # ToDo: define if we want this to plot only one class at a time (delete for loop... not needed)
                for n in range(1):  # leave out the background class
                    channel_data = f[subtomo_h5_internal_path][n + class_number,
                                   overlap:lengths[0] + overlap,
                                   overlap:lengths[1] + overlap,
                                   overlap:lengths[2] + overlap]
                    print("channel ", n, ", min, max = ", np.min(channel_data),
                          np.max(channel_data))
                    internal_subtomo_data += channel_data
                    # ToDo bring back to?:
                    # for n in range(channels - 1):  # leave out the background class
                    #     channel_data = f[subtomo_h5_internal_path][n + 1,
                    #                    overlap:lengths[0] + overlap,
                    #                    overlap:lengths[1] + overlap,
                    #                    overlap:lengths[2] + overlap]
                    #     print("channel ", n, ", min, max = ", np.min(channel_data),
                    #           np.max(channel_data))
                    #     internal_subtomo_data += channel_data
            else:
                internal_subtomo_data = f[subtomo_h5_internal_path][0,
                                        overlap:lengths[0] + overlap,
                                        overlap:lengths[1] + overlap,
                                        overlap:lengths[2] + overlap]
            tomo_data[start_corner[0]: end_corner[0],
            start_corner[1]: end_corner[1],
            start_corner[2]: end_corner[2]] = internal_subtomo_data
            print("internal_subtomo_data = ",
                  internal_subtomo_data.shape)
    write_dataset_hdf(output_path, tomo_data)
    print("right before deleting the maximum is", np.max(tomo_data))
    del tomo_data


def write_dataset_from_subtomos_with_overlap_dice_multiclass(output_path,
                                                             subtomo_path,
                                                             output_shape,
                                                             subtomo_shape,
                                                             subtomos_internal_path,
                                                             class_number,
                                                             overlap):
    output_shape_with_overlap = output_shape  # [dim + overlap_thickness for
    # dim in
    # output_shape]
    print("The output shape is", output_shape_with_overlap)
    tomo_data = np.zeros(output_shape_with_overlap)

    internal_subtomo_shape = tuple([subtomo_dim - 2 * overlap for
                                    subtomo_dim in subtomo_shape])
    with h5py.File(subtomo_path, 'r') as f:
        for subtomo_name in list(f[subtomos_internal_path]):
            subtomo_center = subtomos.get_coord_from_name(subtomo_name)
            start_corner, end_corner, lengths = subtomos.get_subtomo_corners(
                output_shape,
                internal_subtomo_shape,
                subtomo_center)
            overlap_shift = overlap * np.array([1, 1, 1])
            start_corner -= overlap_shift
            end_corner -= overlap_shift
            subtomo_h5_internal_path = join(subtomos_internal_path,
                                            subtomo_name)
            channels = f[subtomo_h5_internal_path][:].shape[0]
            internal_subtomo_data = np.zeros(lengths)
            if channels > 1:
                # ToDo: define if we want this to plot only one class at a time (delete for loop... not needed)
                for n in range(1):  # leave out the background class
                    channel_data = f[subtomo_h5_internal_path][class_number,
                                   overlap:lengths[0] + overlap,
                                   overlap:lengths[1] + overlap,
                                   overlap:lengths[2] + overlap]
                    print("channel ", n, ", min, max = ", np.min(channel_data),
                          np.max(channel_data))
                    internal_subtomo_data += channel_data
                    # ToDo bring back to?:
                    # for n in range(channels - 1):  # leave out the background class
                    #     channel_data = f[subtomo_h5_internal_path][n + 1,
                    #                    overlap:lengths[0] + overlap,
                    #                    overlap:lengths[1] + overlap,
                    #                    overlap:lengths[2] + overlap]
                    #     print("channel ", n, ", min, max = ", np.min(channel_data),
                    #           np.max(channel_data))
                    #     internal_subtomo_data += channel_data
            else:
                internal_subtomo_data = f[subtomo_h5_internal_path][0,
                                        overlap:lengths[0] + overlap,
                                        overlap:lengths[1] + overlap,
                                        overlap:lengths[2] + overlap]
            tomo_data[start_corner[0]: end_corner[0],
            start_corner[1]: end_corner[1],
            start_corner[2]: end_corner[2]] = internal_subtomo_data
            print("internal_subtomo_data = ",
                  internal_subtomo_data.shape)
    write_dataset_hdf(output_path, tomo_data)
    print("right before deleting the maximum is", np.max(tomo_data))
    del tomo_data


def write_dataset_from_subtomos_with_overlap_multiclass_exponentiating(
        output_path,
        subtomo_path,
        output_shape,
        subtomo_shape,
        subtomos_internal_path,
        overlap):
    output_shape_with_overlap = output_shape  # [dim + overlap_thickness for
    # dim in
    # output_shape]
    print("The actual output shape is", output_shape_with_overlap)
    tomo_data = np.zeros(output_shape_with_overlap)

    internal_subtomo_shape = tuple([subtomo_dim - 2 * overlap for
                                    subtomo_dim in subtomo_shape])
    with h5py.File(subtomo_path, 'r') as f:
        for subtomo_name in list(f[subtomos_internal_path]):
            subtomo_center = subtomos.get_coord_from_name(subtomo_name)
            start_corner, end_corner, lengths = subtomos.get_subtomo_corners(
                output_shape,
                internal_subtomo_shape,
                subtomo_center)
            overlap_shift = overlap * np.array([1, 1, 1])
            start_corner -= overlap_shift
            end_corner -= overlap_shift
            subtomo_h5_internal_path = join(subtomos_internal_path,
                                            subtomo_name)
            channels = f[subtomo_h5_internal_path][:].shape[0]
            internal_subtomo_data = np.zeros(lengths)
            for n in range(channels - 1):  # leave out the background class
                channel_data = f[subtomo_h5_internal_path][n + 1,
                               overlap:lengths[0] + overlap,
                               overlap:lengths[1] + overlap,
                               overlap:lengths[2] + overlap]
                print("channel ", n, ", min, max = ", np.min(channel_data),
                      np.max(channel_data))
                internal_subtomo_data += np.exp(channel_data)

            tomo_data[start_corner[0]: end_corner[0],
            start_corner[1]: end_corner[1],
            start_corner[2]: end_corner[2]] = internal_subtomo_data
            print("internal_subtomo_data = ",
                  internal_subtomo_data.shape)
    write_dataset_hdf(output_path, tomo_data)
    print("right before deleting the maximum is", np.max(tomo_data))
    del tomo_data


def write_subtomograms_from_dataset(output_path, padded_dataset,
                                    window_centers, crop_shape):
    with h5py.File(output_path, 'w') as f:
        for window_center in window_centers:
            subtomo_name = "subtomo_{0}".format(str(window_center))
            subtomo_h5_internal_path = join(h5_internal_paths.RAW_SUBTOMOGRAMS,
                                            subtomo_name)
            subtomo_data = crop_window_around_point(input=padded_dataset,
                                                    crop_shape=crop_shape,
                                                    window_center=window_center)
            f[subtomo_h5_internal_path] = subtomo_data


def write_joint_raw_and_labels_subtomograms(output_path: str,
                                            padded_raw_dataset: np.array,
                                            padded_labels_dataset: np.array,
                                            label_name: str,
                                            window_centers: list,
                                            crop_shape: tuple):
    with h5py.File(output_path, 'w') as f:
        for window_center in window_centers:
            # print("window_center", window_center)
            subtomo_name = "subtomo_{0}".format(str(window_center))
            subtomo_raw_h5_internal_path = join(
                h5_internal_paths.RAW_SUBTOMOGRAMS,
                subtomo_name)
            subtomo_raw_data = crop_window_around_point(
                input=padded_raw_dataset,
                crop_shape=crop_shape,
                window_center=window_center)

            subtomo_label_h5_internal_path = join(
                h5_internal_paths.LABELED_SUBTOMOGRAMS, label_name)
            subtomo_label_h5_internal_path = join(
                subtomo_label_h5_internal_path,
                subtomo_name)

            subtomo_label_data = crop_window_around_point(
                input=padded_labels_dataset,
                crop_shape=crop_shape,
                window_center=window_center)
            if np.max(subtomo_label_data) > 0.5:
                f[subtomo_raw_h5_internal_path] = subtomo_raw_data
                f[subtomo_label_h5_internal_path] = subtomo_label_data
            else:
                print("subtomo ", subtomo_name, "discarded")
    return


def write_joint_raw_and_labels_subtomograms_dice_multiclass(
        output_path: str,
        padded_raw_dataset: np.array,
        padded_labels_dataset_list: list,  # list of padded labeled datasets
        segmentation_names: list,
        window_centers: list,
        crop_shape: tuple):
    with h5py.File(output_path, 'w') as f:
        for window_center in window_centers:
            print("window_center", window_center)
            subtomo_name = "subtomo_{0}".format(str(window_center))
            subtomo_raw_h5_internal_path = join(
                h5_internal_paths.RAW_SUBTOMOGRAMS,
                subtomo_name)
            subtomo_raw_data = crop_window_around_point(
                input=padded_raw_dataset,
                crop_shape=crop_shape,
                window_center=window_center)

            subtomo_label_data_list = []
            subtomo_label_h5_internal_path_list = []
            segmentation_max = 0
            for label_name, padded_labels_dataset in zip(segmentation_names,
                                                         padded_labels_dataset_list):
                subtomo_label_data = crop_window_around_point(
                    input=padded_labels_dataset,
                    crop_shape=crop_shape,
                    window_center=window_center)
                subtomo_label_data_list += [subtomo_label_data]
                print("subtomo_max = ", np.max(subtomo_label_data))
                subtomo_label_h5_internal_path = join(
                    h5_internal_paths.LABELED_SUBTOMOGRAMS, label_name)
                subtomo_label_h5_internal_path = join(
                    subtomo_label_h5_internal_path,
                    subtomo_name)
                subtomo_label_h5_internal_path_list += [
                    subtomo_label_h5_internal_path]
                segmentation_max = np.max(
                    [segmentation_max, np.max(subtomo_label_data)])
            if segmentation_max > 0.5:
                f[subtomo_raw_h5_internal_path] = subtomo_raw_data
                for subtomo_label_h5_internal_path, subtomo_label_data in zip(
                        subtomo_label_h5_internal_path_list,
                        subtomo_label_data_list):
                    f[subtomo_label_h5_internal_path] = subtomo_label_data
            else:
                print("subtomo ", subtomo_name, "discarded")
    return


def write_segmented_data(data_path: str, output_segmentation: np.array,
                         label_name: str) -> np.array:
    with h5py.File(data_path, 'a') as f:
        for subtomo_indx, subtomo_name in enumerate(
                list(f[h5_internal_paths.RAW_SUBTOMOGRAMS])):
            segmented_subtomo_path = join(
                h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS,
                label_name)
            subtomo_h5_internal_path = join(segmented_subtomo_path,
                                            subtomo_name)
            f[subtomo_h5_internal_path] = output_segmentation[subtomo_indx, :,
                                          :, :, :]


def segment_and_write(data_path: str, model: UNet, label_name: str):
    with h5py.File(data_path, 'a') as data_file:
        print(h5_internal_paths.RAW_SUBTOMOGRAMS)
        print(list(data_file[h5_internal_paths.RAW_SUBTOMOGRAMS]))
        for subtomo_name in list(data_file[h5_internal_paths.RAW_SUBTOMOGRAMS]):
            subtomo_h5_internal_path = join(h5_internal_paths.RAW_SUBTOMOGRAMS,
                                            subtomo_name)
            subtomo_data = np.array([data_file[subtomo_h5_internal_path][:]])
            subtomo_data = subtomo_data[:, None]
            print("subtomo_shape ", subtomo_data.shape)
            print("segmenting ", subtomo_name)
            segmented_data = model(torch.from_numpy(subtomo_data))
            segmented_data = segmented_data.detach().numpy()
            _write_segmented_subtomo_data(data_file=data_file,
                                          segmented_data=segmented_data,
                                          label_name=label_name,
                                          subtomo_name=subtomo_name)

    return


def _write_segmented_subtomo_data(data_file: h5py.File,
                                  segmented_data: np.array,
                                  label_name: str,
                                  subtomo_name: str):
    subtomo_h5_internal_path = join(
        h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS,
        label_name)
    subtomo_h5_internal_path = join(subtomo_h5_internal_path,
                                    subtomo_name)
    print(subtomo_h5_internal_path)
    data_file[subtomo_h5_internal_path] = segmented_data[0, :, :, :, :]
    print("data shape =", segmented_data[0, :, :, :, :].shape)


def write_hdf_particles_from_motl(path_to_motl: str,
                                  hdf_output_path: str,
                                  output_shape: tuple,
                                  sphere_radius=8,
                                  values_in_motl=True,
                                  number_of_particles=None,
                                  z_shift=0,
                                  particle_classes=[1],
                                  particles_in_tom_format=True):
    _, file_extension = os.path.splitext(path_to_motl)
    print("The motive list has extension ", file_extension)
    assert file_extension == ".csv" or file_extension == ".em" or file_extension == ".txt"

    if file_extension == ".csv" or file_extension == ".em":
        if file_extension == ".csv":
            motive_list = read_motl_from_csv(path_to_motl)
            if isinstance(number_of_particles, int):
                motive_list = motive_list[:number_of_particles]
                print("Only", str(number_of_particles),
                      " particles in the motive list will be pasted.")
            else:
                print("All particles in the motive list will be pasted.")
                # Already in x,y,z format:
            if particles_in_tom_format:
                coordinates = [
                    np.array([int(row[9]) + z_shift, int(row[8]), int(row[7])])
                    for
                    row in motive_list]
            else:
                coordinates = [
                    np.array([int(row[7]) + z_shift, int(row[8]), int(row[9])])
                    for
                    row in motive_list]
            if values_in_motl:
                score_values = [row[0] for row in motive_list]
            else:
                score_values = np.ones(len(motive_list))
                print("The map will be binary.")
        else:
            _, motive_list = load_em_motl(path_to_emfile=path_to_motl)
            if isinstance(number_of_particles, int):
                motive_list = motive_list[:number_of_particles]
                print("Only", str(number_of_particles),
                      " particles in the motive list will be pasted.")
            else:
                print("All particles in the motive list will be pasted.")
            coordinates = extract_coordinates_from_em_motl(motive_list)

            if particles_in_tom_format:
                print("coordinates already in tom format")
                coordinates = [[int(p[2]) + z_shift, int(p[1]), int(p[0])] for p
                               in coordinates]
            else:
                print("transforming coordinates to tom format")
                coordinates = [[int(p[2]) + z_shift, int(p[1]), int(p[0])] for p
                               in coordinates]

            score_values = np.ones(len(coordinates))

        predicted_dataset = np.zeros(output_shape)
        for center, value in zip(coordinates, score_values):
            paste_sphere_in_dataset(dataset=predicted_dataset,
                                    radius=sphere_radius,
                                    value=value, center=center)

        write_dataset_hdf(output_path=hdf_output_path,
                          tomo_data=predicted_dataset)
    elif file_extension == ".txt":
        motive_list = read_shrec_motl(path_to_motl)
        predicted_dataset = np.zeros(output_shape)
        for counter, particle_class in enumerate(particle_classes):
            if isinstance(particle_class, int):
                if values_in_motl:
                    value = counter + 1
                else:
                    value = 1
                coordinates = extract_coordinates_from_txt_shrec(
                    motive_list=motive_list, particle_class=particle_class)
                coordinates = [p + np.array([z_shift, 0, 0]) for p in
                               coordinates]
                for center in coordinates:
                    paste_sphere_in_dataset(dataset=predicted_dataset,
                                            radius=sphere_radius,
                                            value=value,
                                            center=center)
            elif isinstance(particle_class, str):
                particle_class_number = particle_dict[particle_class]['class']
                if values_in_motl:
                    value = counter + 1
                    print("The value assigned to", particle_class, "is", value)
                else:
                    value = 1
                    print("The value assigned to", particle_class, "is", value)

                sphere_radius = particle_dict[particle_class]['radius']
                print("The radius in pixels of", particle_class, "is",
                      sphere_radius)
                coordinates = extract_coordinates_from_txt_shrec(
                    motive_list=motive_list,
                    particle_class=particle_class_number)
                print("len(coordinates) =", len(coordinates))
                coordinates = [p + np.array([z_shift, 0, 0]) for p in
                               coordinates]
                print("coordinates[0] = ", coordinates[0])
                for center in coordinates:
                    paste_sphere_in_dataset(dataset=predicted_dataset,
                                            radius=sphere_radius,
                                            value=value,
                                            center=center)
            else:
                print("particle classes should be either integers or strings.")
        write_dataset_hdf(output_path=hdf_output_path,
                          tomo_data=predicted_dataset)

    else:
        print("The motive list is not written in a valid file format.")

    return


def split_and_write_h5_partition(h5_partition_data_path: str,
                                 h5_train_patition_path: str,
                                 h5_test_patition_path: str,
                                 split: float,
                                 label_name="particles",
                                 shuffle=True) -> tuple:
    with h5py.File(h5_partition_data_path, 'r') as f:
        raw_subtomo_names = list(f[h5_internal_paths.RAW_SUBTOMOGRAMS])
        if shuffle:
            random.shuffle(raw_subtomo_names)
        else:
            print("Splitting sets without shuffling")
        split = int(split * len(raw_subtomo_names))
        with h5py.File(h5_train_patition_path, "w") as f_train:
            for subtomo_name in raw_subtomo_names[:split]:
                raw_subtomo_h5_internal_path \
                    = join(h5_internal_paths.RAW_SUBTOMOGRAMS, subtomo_name)
                data_raw_train = f[raw_subtomo_h5_internal_path][:]

                labels_subtomo_h5_internal_path = join(
                    h5_internal_paths.LABELED_SUBTOMOGRAMS, label_name)
                labels_subtomo_h5_internal_path = join(
                    labels_subtomo_h5_internal_path,
                    subtomo_name)
                data_label_train = f[labels_subtomo_h5_internal_path][:]

                f_train[raw_subtomo_h5_internal_path] = data_raw_train
                f_train[labels_subtomo_h5_internal_path] = data_label_train
        with h5py.File(h5_test_patition_path, "w") as f_test:
            for subtomo_name in raw_subtomo_names[split:]:
                raw_subtomo_h5_internal_path \
                    = join(h5_internal_paths.RAW_SUBTOMOGRAMS, subtomo_name)
                data_raw_test = f[raw_subtomo_h5_internal_path][:]

                labels_subtomo_h5_internal_path = join(
                    h5_internal_paths.LABELED_SUBTOMOGRAMS, label_name)
                labels_subtomo_h5_internal_path = join(
                    labels_subtomo_h5_internal_path,
                    subtomo_name)
                data_label_test = f[labels_subtomo_h5_internal_path][:]

                f_test[raw_subtomo_h5_internal_path] = data_raw_test
                f_test[labels_subtomo_h5_internal_path] = data_label_test

    return


def split_and_write_h5_partition_dice_multi_class(h5_partition_data_path: str,
                                                  h5_train_patition_path: str,
                                                  h5_test_patition_path: str,
                                                  split: int,
                                                  segmentation_names: list,
                                                  shuffle=True) -> tuple:
    with h5py.File(h5_partition_data_path, 'r') as f:
        raw_subtomo_names = list(f[h5_internal_paths.RAW_SUBTOMOGRAMS])
        if shuffle:
            random.shuffle(raw_subtomo_names)
        else:
            print("Splitting sets without shuffling")
        with h5py.File(h5_train_patition_path, "w") as f_train:
            for subtomo_name in raw_subtomo_names[:split]:
                raw_subtomo_h5_internal_path \
                    = join(h5_internal_paths.RAW_SUBTOMOGRAMS, subtomo_name)
                data_raw_train = f[raw_subtomo_h5_internal_path][:]
                f_train[raw_subtomo_h5_internal_path] = data_raw_train
                for label_name in segmentation_names:
                    labels_subtomo_h5_internal_path = join(
                        h5_internal_paths.LABELED_SUBTOMOGRAMS, label_name)
                    labels_subtomo_h5_internal_path = join(
                        labels_subtomo_h5_internal_path,
                        subtomo_name)
                    data_label_train = f[labels_subtomo_h5_internal_path][:]
                    f_train[labels_subtomo_h5_internal_path] = data_label_train

        with h5py.File(h5_test_patition_path, "w") as f_test:
            for subtomo_name in raw_subtomo_names[split:]:
                raw_subtomo_h5_internal_path = join(
                    h5_internal_paths.RAW_SUBTOMOGRAMS, subtomo_name)
                data_raw_test = f[raw_subtomo_h5_internal_path][:]
                f_test[raw_subtomo_h5_internal_path] = data_raw_test
                for label_name in segmentation_names:
                    labels_subtomo_h5_internal_path = join(
                        h5_internal_paths.LABELED_SUBTOMOGRAMS, label_name)
                    labels_subtomo_h5_internal_path = join(
                        labels_subtomo_h5_internal_path,
                        subtomo_name)
                    data_label_test = f[labels_subtomo_h5_internal_path][:]
                    f_test[labels_subtomo_h5_internal_path] = data_label_test
    return
