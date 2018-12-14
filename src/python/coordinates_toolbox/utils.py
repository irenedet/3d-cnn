from typing import Tuple

import numpy as np

from src.python.naming.particles import create_particle_file_name


def rearrange_hdf_coordinates(p: list) -> tuple:
    """
    Function that rearranges a tuple from hdf load in the x,y,z
    order.
    :param p: a point comming from hdf.load_dataset with format [z, y, x]
    :return: (x, y, z)
    """
    return p[2], p[1], p[0]


def shift_coordinates(coordinates: np.array, origin: tuple) -> np.array:
    """ dim_x, dim_y, dim_z """
    m0, m1, m2 = origin
    coordinates_shifted = np.array(
        [[p[0] - m0, p[1] - m1, p[2] - m2] for p in coordinates])
    return coordinates_shifted


def _boxing2D(dataset: np.array, point: Tuple, size: int) -> np.array:
    ds = int(0.5 * size)
    _, ds_side_length, _ = dataset.shape
    x, y, z = point
    x = int(x)
    y = int(y)
    z = int(z)
    if (x - ds) >= 0 and (y - ds) >= 0 and (x + ds) < ds_side_length and (
                y + ds) < ds_side_length:
        box = dataset[z, y - ds:y + ds, x - ds:x + ds]
        return box
    else:
        print("Particle " + str(
            point) + " is too close to the border of this data set.")
        return []


def store_imgs_as_txt(dest_folder_path: str,
                      dataset: np.array,
                      particle_coords: np.array,
                      sampling_points_indices: list,
                      box_size: int):
    img_number = 0
    for sampling_point_indx in sampling_points_indices:
        img_number += 1
        particle_point = particle_coords[sampling_point_indx, :]
        box2D = _boxing2D(dataset, particle_point, box_size)
        if len(box2D):
            img = _boxing2D(dataset, particle_point, box_size)
            _store_as_txt(folder_path=dest_folder_path,
                          img=img,
                          coord_indx=sampling_point_indx,
                          img_number=img_number)
    return


def _store_as_txt(folder_path: str, img: np.array, coord_indx: int,
                  img_number: int):
    file_name = create_particle_file_name(folder_path, img_number, coord_indx,
                                          'txt')
    np.savetxt(file_name, img, fmt='%10.5f')
    return


def extract_coordinates_from_motl(motl: np.array) -> np.array:
    return np.array(motl[0, :, 7:10])



