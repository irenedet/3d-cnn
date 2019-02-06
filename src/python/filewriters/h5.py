import h5py
import numpy as np
from os.path import join

import torch

from src.python.naming import h5_internal_paths
from src.python.coordinates_toolbox import subtomos
from src.python.tensors.actions import crop_window_around_point
from src.python.pytorch_cnn.classes.unet_new import UNet


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
        for subtomo_name in list(data_file[h5_internal_paths.RAW_SUBTOMOGRAMS]):
            subtomo_h5_internal_path = join(h5_internal_paths.RAW_SUBTOMOGRAMS,
                                            subtomo_name)
            subtomo_data = np.array([data_file[subtomo_h5_internal_path][:]])
            subtomo_data = subtomo_data[:, None]
            segmentated_data = model(torch.from_numpy(subtomo_data))
            segmentated_data = segmentated_data.detach().numpy()
            _write_segmented_subtomo_data(data_file=data_file,
                                          segmented_data=segmentated_data,
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
    data_file[subtomo_h5_internal_path] = segmented_data[0, :, :, :, :]
    return
