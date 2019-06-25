import re
from typing import List

import numpy as np
import h5py
from src.python.naming import h5_internal_paths


def get_coord_from_name(subtomo_name):
    return [int(val) for val in re.findall(r'\d+', subtomo_name)]


def get_subtomo_corners(output_shape: tuple, subtomo_shape: tuple,
                        subtomo_center: tuple) -> tuple:
    subtomo_l1radius = subtomo_shape[0] // 2, subtomo_shape[1] // 2, \
                       subtomo_shape[2] // 2
    start_corners = [int(center_dim) - int(subtomo_dim) for
                     center_dim, subtomo_dim
                     in zip(subtomo_center, subtomo_l1radius)]
    end_corners = [center_dim + subtomo_dim for center_dim, subtomo_dim
                   in zip(subtomo_center, subtomo_l1radius)]
    end_corners = [int(np.min((end_point, tomo_dim))) for end_point, tomo_dim
                   in zip(end_corners,
                          output_shape)]
    side_lengths = [int(end - start) for end, start in
                    zip(end_corners, start_corners)]
    return start_corners, end_corners, side_lengths


def get_particle_coordinates_grid_with_overlap(dataset_shape, shape_to_crop_zyx,
                                               overlap_thickness):
    dataset_without_overlap_shape = [tomo_dim - 2 * overlap_thickness for
                                     tomo_dim in dataset_shape]
    internal_shape_to_crop_zyx = [dim - 2 * overlap_thickness for
                                  dim in shape_to_crop_zyx]

    particle_coordinates = get_particle_coordinates_grid(
        dataset_without_overlap_shape,
        internal_shape_to_crop_zyx)
    overlap_shift = overlap_thickness * np.array([1, 1, 1])
    particle_coordinates_with_overlap = [point + overlap_shift
                                         for point in particle_coordinates]
    return particle_coordinates_with_overlap


def get_particle_coordinates_grid(dataset_shape, shape_to_crop_zyx):
    particle_coordinates = []
    nz_coords, ny_coords, nx_coords = [tomo_dim // box_size for
                                       tomo_dim, box_size in
                                       zip(dataset_shape, shape_to_crop_zyx)]
    for z in range(nz_coords):
        for y in range(ny_coords):
            for x in range(nx_coords):
                particle_coordinates += [
                    np.array(shape_to_crop_zyx) * np.array([z, y, x])
                    + np.array(shape_to_crop_zyx) // 2]
    return particle_coordinates


def read_subtomo_names(subtomo_file_path):
    with h5py.File(subtomo_file_path, 'r') as f:
        return list(f[h5_internal_paths.RAW_SUBTOMOGRAMS])


def get_subtomo_corners_within_dataset(dataset_shape: tuple or List[int],
                                       subtomo_shape: tuple or List[int],
                                       center: tuple or List[
                                           int]) -> tuple:
    subtomo_l1radius = subtomo_shape[0] // 2, subtomo_shape[1] // 2, \
                       subtomo_shape[2] // 2
    start_corners = [center_dim - subtomo_dim for center_dim, subtomo_dim
                     in zip(center, subtomo_l1radius)]
    end_corners = [center_dim + subtomo_dim for center_dim, subtomo_dim
                   in zip(center, subtomo_l1radius)]
    end_corners = [np.min((end_point, tomo_dim)) for end_point, tomo_dim
                   in zip(end_corners,
                          dataset_shape)]

    start_corners = [np.max([corner, 0]) for corner in start_corners]
    end_corners = [np.min([corner, data_sh]) for corner, data_sh in
                   zip(end_corners, dataset_shape)]
    side_lengths = [end - start for start, end in
                    zip(start_corners, end_corners)]
    return start_corners, end_corners, side_lengths
