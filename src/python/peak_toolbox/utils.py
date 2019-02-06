import csv
from os.path import join

import numpy as np

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
    global_max = np.ndarray.max(dataset)
    global_max_coords = np.where(dataset == global_max)
    coordinates_list = [(global_max_coords[0][0], global_max_coords[1][0],
                         global_max_coords[2][0])]

    list_of_maxima = [global_max]
    list_of_maxima_coords = coordinates_list

    for n in range(numb_peaks):
        next_max, coordinates_list, flag = _get_next_max(dataset,
                                                         coordinates_list,
                                                         radius,
                                                         numb_peaks)
        if "overloaded" == flag:
            print("overloaded, reached a level with no more local maxima")
            print("maxima in subtomo:", n)
            return list_of_maxima, list_of_maxima_coords
        elif next_max == -100:
            print("maxima in subtomo:", n)
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


def extract_motl_coordinates_and_score_values(motl: list):
    coordinates = [np.array([row[7], row[8], row[9]]) for row in
                   motl]
    score_values = [row[0] for row in motl]
    return score_values, coordinates