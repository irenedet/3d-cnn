import csv
import os
from os.path import join

import numpy as np
from src.python.coordinates_toolbox.utils import \
    extract_coordinates_and_values_from_em_motl
from src.python.filereaders.csv import read_motl_from_csv
from src.python.filereaders.em import load_em_motl
from src.python.osactions.filesystem import create_dir


def _generate_unit_particle(radius: int):
    unit_particle = [(0, 0, 0)]
    for i in range(radius):
        for j in range(radius):
            for k in range(radius):
                if np.sqrt(i ** 2 + j ** 2 + k ** 2) <= radius:
                    unit_particle += [(i, j, k), (-i, j, k), (i, -j, k),
                                      (i, j, -k), (-i, -j, k), (-i, j, -k),
                                      (i, -j, -k), (-i, -j, -k)]
    return unit_particle


def paste_sphere_in_dataset(dataset: np.array, radius: int, value: float,
                            center: tuple) -> np.array:
    dataset_dimensions = dataset.shape
    unit_particle = _generate_unit_particle(radius)
    particle = [
        (center[0] + delta_p[0], center[1] + delta_p[1], center[2] + delta_p[2])
        for delta_p in unit_particle]
    for coord in particle:
        if (coord[0] < dataset_dimensions[0]) and (
                    coord[1] < dataset_dimensions[1]) and (
                    coord[2] < dataset_dimensions[2]) and (0 <= coord[0]) and (
                    0 <= coord[1]) and (0 <= coord[2]):
            dataset[coord[0], coord[1], coord[2]] = value
    return dataset


def _get_next_max(dataset: np.array, coordinates_list: list, radius: int,
                  numb_peaks: int) -> tuple:
    dataset_dimensions = dataset.shape
    unit_particle = _generate_unit_particle(radius)
    if len(coordinates_list) < numb_peaks:
        for p in coordinates_list:
            particle = [
                (p[0] + delta_p[0], p[1] + delta_p[1], p[2] + delta_p[2])
                for delta_p in unit_particle]
            for coord in particle:
                if (coord[0] < dataset_dimensions[0]) and (
                            coord[1] < dataset_dimensions[1]) and (
                            coord[2] < dataset_dimensions[2]) and (
                            0 <= coord[0]) and (
                            0 <= coord[1]) and (0 <= coord[2]):
                    dataset[coord[0]][coord[1]][coord[2]] = -100
        next_max = np.ndarray.max(dataset)
        next_max_coords = np.where(next_max == dataset)
        next_max_coords_list = []
        n = len(next_max_coords[0])
        for next_p in range(n):
            next_max_coords_list += [(next_max_coords[0][next_p],
                                      next_max_coords[1][next_p],
                                      next_max_coords[2][next_p])]
        flag = "not_overloaded"
    else:
        next_max = 0
        next_max_coords_list = []
        flag = "overloaded"
    return next_max, next_max_coords_list, flag


def extract_peaks(dataset: np.array, numb_peaks: int, radius: int):
    # Todo: instead of -100 use global_min - 1...
    # global_min = np.min(dataset)
    global_max = np.ndarray.max(dataset)
    global_max_coords = np.where(dataset == global_max)
    print(global_max_coords)
    coordinates_list = [(global_max_coords[0][0], global_max_coords[1][0],
                         global_max_coords[2][0])]
    print(coordinates_list)
    list_of_maxima = [global_max]
    list_of_maxima_coords = coordinates_list

    for n in range(numb_peaks):
        next_max, coordinates_list, flag = _get_next_max(dataset,
                                                         coordinates_list,
                                                         radius,
                                                         numb_peaks)
        if "overloaded" == flag:
            print("overloaded, reached a level with no more local maxima")
            print("maxima in subtomo:", len(list_of_maxima_coords))
            return list_of_maxima, list_of_maxima_coords
        elif next_max == -100:
            print("maxima in subtomo:", list_of_maxima_coords)
            return list_of_maxima, list_of_maxima_coords
        else:
            list_of_maxima += [next_max for _ in coordinates_list]
            list_of_maxima_coords += coordinates_list
    return list_of_maxima, list_of_maxima_coords


def write_csv_motl(list_of_maxima: list, list_of_maxima_coords: list,
                   motl_output_dir: str):
    create_dir(motl_output_dir)

    numb_peaks = len(list_of_maxima)
    motl_file_name = join(motl_output_dir, 'motl_' + str(numb_peaks) + '.csv')

    with open(motl_file_name, 'w', newline='') as csv_file:
        motl_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                 quoting=csv.QUOTE_MINIMAL)

        for val, p in zip(list_of_maxima, list_of_maxima_coords):
            motl_writer.writerow([str(val) + ',' + str(p[1]) + ',' + str(
                p[2]) + ',' + str(p[0]) + ',0,0,0,' + str(p[1]) + ',' + str(
                p[2]) + ',' + str(p[0]) + ',0,0,0,0,0,0,0,0,0,1'])
    print("motive list writen in ", motl_file_name)
    return


def extract_motl_coordinates_and_score_values(motl: list) -> tuple:
    coordinates = [np.array([row[7], row[8], row[9]]) for row in
                   motl]
    score_values = [row[0] for row in motl]
    return score_values, coordinates


def _generate_horizontal_disk(radius: int, thickness: int) -> list:
    disk = []
    for i in range(radius):
        for j in range(radius):
            for k in range(thickness // 2):
                if np.sqrt(i ** 2 + j ** 2) <= radius:
                    disk += [(k, i, j), (k, -i, j), (k, i, -j), (k, -i, -j)]
                    if k > 0:
                        disk += [(-k, i, j), (-k, -i, j), (-k, i, -j),
                                 (-k, -i, -j)]
                        # disk += [(i, j, k), (-i, j, k), (i, -j, k), (-i, -j, k)]
                        # if k > 0:
                        #     disk += [(i, j, -k), (-i, j, -k), (i, -j, -k),
                        #              (-i, -j, -k)]

    return disk


def paste_rotated_disk(dataset: np.array, center: tuple, radius: int,
                       thickness: int,
                       ZXZ_angles: tuple):
    cx, cy, cz = center
    psi, theta, sigma = ZXZ_angles
    to_radians = lambda theta: theta * np.pi / 180
    #
    psi = to_radians(psi)
    theta = to_radians(theta)
    sigma = to_radians(sigma)

    rot_z = lambda psi: np.array(
        [[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0],
         [0, 0, 1]])

    rot_x = lambda psi: np.array([[1, 0, 0], [0, np.cos(psi), -np.sin(psi)],
                                  [0, np.sin(psi), np.cos(psi)]])

    # To fit hdf coordinate system:
    hdf_coords = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    swap_coords = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    ZXZ_matrix = rot_z(psi).dot(rot_x(theta))
    ZXZ_matrix = ZXZ_matrix.dot(rot_z(sigma))
    print(ZXZ_matrix)
    ZXZ_matrix = hdf_coords.dot(ZXZ_matrix)
    ZXZ_matrix = ZXZ_matrix.dot(swap_coords)

    disk = _generate_horizontal_disk(radius, thickness)
    new_disk = []
    for point in disk:
        new_disk += [ZXZ_matrix.dot(np.array(point))]
    for point in new_disk:
        i, j, k = point
        i = int(i)
        j = int(j)
        k = int(k)
        dataset[i + cx, j + cy, k + cz] = 1

    return dataset


def read_motl_coordinates_and_values(path_to_motl: str):
    _, motl_extension = os.path.splitext(path_to_motl)
    if motl_extension == ".em":
        print("motl in .em format")
        header, motl = load_em_motl(path_to_emfile=path_to_motl)
        motl_values, motl_coords = extract_coordinates_and_values_from_em_motl(
            motl)
        return motl_values, motl_coords
    elif motl_extension == ".csv":
        print("motl in .csv format")
        motl = read_motl_from_csv(path_to_motl)
        motl_values, motl_coords = extract_motl_coordinates_and_score_values(
            motl)
        return motl_values, np.array(motl_coords)
    else:
        print("motl clean should be in a valid format .em or .csv")


def union_of_motls(path_to_motl_1: str, path_to_motl_2: str):
    values_1, coordinates_1 = read_motl_coordinates_and_values(
        path_to_motl=path_to_motl_1)
    values_2, coordinates_2 = read_motl_coordinates_and_values(
        path_to_motl=path_to_motl_2)
    coordinates = np.concatenate((coordinates_1, coordinates_2), axis=0)
    values = list(values_1) + list(values_2)
    return values, coordinates
