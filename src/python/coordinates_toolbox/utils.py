from typing import Tuple

import numpy as np

from src.python.naming.particles import create_particle_file_name


def to_tom_coordinate_system(p: list) -> np.array:
    """
    Function that rearranges a list of coordinates from the python hdf load
    coordinates system into the tom coordinate system.
    :param p: a point coming from hdf.load_dataset with format [z, y, x]
    :return: [x, y, z]
    """
    return np.array([p[2], p[1], p[0]])


def invert_tom_coordinate_system(p: list) -> np.array:
    """
    Function that rearranges a list of coordinates from the python hdf load
    coordinates system into the tom coordinate system.
    :param p: a point coming from hdf.load_dataset with format [z, y, x]
    :return: [x, y, z]
    """
    return np.array([p[2], p[1], p[0]])


def arrange_coordinates_list_by_score(list_of_peak_scores: list,
                                      list_of_peak_coordinates: list) -> tuple:
    take_first_entry = lambda pair: pair[0]
    joint_list = list(zip(list_of_peak_scores, list_of_peak_coordinates))
    joint_list = sorted(joint_list, key=take_first_entry, reverse=1)
    unzipped_list = list(zip(*joint_list))
    list_of_peak_scores, list_of_peak_coordinates = list(
        unzipped_list[0]), list(unzipped_list[1])
    return list_of_peak_scores, list_of_peak_coordinates


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


def extract_coordinates_from_em_motl(motl: np.array) -> np.array:
    return np.array(motl[0, :, 7:10])


def extract_coordinates_and_values_from_em_motl(motl: np.array) -> np.array:
    values = np.array(motl[0, :, 0])
    coordinates = np.array(motl[0, :, 7:10])
    return values, coordinates


def extract_coordinates_from_txt_shrec(motive_list: np.array,
                                       particle_class=1) -> np.array:
    n = motive_list.shape[0]
    pre_coordinates = [np.array(motive_list[index, 1:4]) for index in range(n)
                       if motive_list[index, 0] == particle_class]
    coordinates = [[int(val) for val in point] for point in
                   pre_coordinates]
    del pre_coordinates
    return coordinates


def filtering_duplicate_coords(motl_coords: list, min_peak_distance: int):
    unique_motl_coords = [motl_coords[0]]
    for point in motl_coords[1:]:
        flag = "unique"
        n_point = 0
        while flag == "unique" and n_point < len(unique_motl_coords):
            x = unique_motl_coords[n_point]
            n_point += 1
            if np.linalg.norm(x - point) <= min_peak_distance:
                flag = "repeated"
                # print("repeated point = ", point)
        if flag == "unique":
            unique_motl_coords += [point]
    return unique_motl_coords


def filtering_duplicate_coords_with_values(motl_coords: list,
                                           motl_values: list,
                                           min_peak_distance: int,
                                           preference_by_score=True):
    motl_coords = np.array(motl_coords)
    unique_motl_coords = [motl_coords[0]]
    unique_motl_values = [motl_values[0]]

    for value, point in zip(motl_values[1:], motl_coords[1:]):
        flag = "unique"
        n_point = 0
        while flag == "unique" and n_point < len(unique_motl_coords):
            x = unique_motl_coords[n_point]
            x_val = unique_motl_values[n_point]
            n_point += 1
            if np.linalg.norm(x - point) <= min_peak_distance:
                flag = "repeated"
                if preference_by_score and (x_val < value):
                    unique_motl_coords[n_point] = point
                    unique_motl_values[n_point] = value
        if flag == "unique":
            unique_motl_coords += [point]
            unique_motl_values += [value]
    print("Number of unique coordinates after filtering:",
          len(unique_motl_coords))
    return unique_motl_values, unique_motl_coords
