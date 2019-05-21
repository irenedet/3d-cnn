import csv
import datetime
import re
import time
from os.path import join

import h5py
import numpy as np

from src.python.coordinates_toolbox.utils import \
    filtering_duplicate_coords_with_values, rearrange_hdf_coordinates
from src.python.naming import h5_internal_paths
from src.python.peak_toolbox.subtomos import \
    get_peaks_per_subtomo_with_overlap, \
    get_peaks_per_subtomo_with_overlap_multiclass, \
    get_subtomo_corner_and_side_lengths, shift_coordinates_by_vector

from src.python.peak_toolbox.utils import union_of_motls


def motl_writer(path_to_output_folder: str, list_of_peak_scores: list,
                list_of_peak_coords: list, in_tom_format=False,
                order_by_score=True, motl_name=None):
    """
    Already modified to match em_motl format
    """
    numb_peaks = len(list_of_peak_scores)
    joint_list = list(zip(list_of_peak_scores, list_of_peak_coords))

    if order_by_score:
        print("saving coordinates ordered by decreasing score value")
        joint_list = sorted(joint_list, key=lambda pair: pair[0], reverse=1)
    else:
        print("saving coordinates without sorting by score value")

    if motl_name is None:
        motl_file_name = join(path_to_output_folder,
                              'motl_' + str(numb_peaks) + '.csv')
    else:
        motl_file_name = join(path_to_output_folder, motl_name)
    with open(motl_file_name, 'w', newline='') as csvfile:
        motlwriter = csv.writer(csvfile, delimiter=' ', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        for indx, tuple_val_point in enumerate(joint_list):
            val, point = tuple_val_point
            if in_tom_format:
                x, y, z = point
            else:
                x, y, z = rearrange_hdf_coordinates(point)
            coordinate_in_tom_format = str(x) + ',' + str(y) + ',' + str(z)
            angles = ',0,0,0,'
            blank_columns = ',0,0,'
            tail = ',0,0,0,0,0,0,0,0,0,1'
            row = str(val) + blank_columns + str(
                indx) + angles + coordinate_in_tom_format + tail
            motlwriter.writerow([row])
    print("The motive list has been writen in", motl_file_name)
    return motl_file_name


def _write_table_header(directory_path: str, param: str,
                        table_writer: csv.writer):
    now = datetime.datetime.now()
    table_writer.writerow([str(now)])
    table_writer.writerow(["From jobs in " + directory_path])
    table_writer.writerow(["CONTENTS"])
    table_writer.writerow(["_job_name"])
    table_writer.writerow(["_K"])
    table_writer.writerow(["_" + param])
    table_writer.writerow(["_classes"])
    table_writer.writerow(["_auPRC"])
    return


def write_jobs_table(directory_path: str, table_name: str, param: str,
                     star_files: list, jobs_statistics_dict: dict):
    table_file_path = join(directory_path, table_name)
    with open(table_file_path, 'w') as csvfile:
        table_writer = csv.writer(csvfile, delimiter=' ',
                                  quotechar='|', quoting=csv.QUOTE_MINIMAL)
        _write_table_header(directory_path, param, table_writer)
        for job_parameters in star_files:
            job_path, k, param_value, classes = job_parameters
            classes = set(classes)
            job_name = re.findall(r"(job\d\d\d)", job_path)[0]
            _, _, _, au_prc, _ = jobs_statistics_dict[job_name]
            row = [job_name, k, param_value, classes, au_prc]
            table_writer.writerow(row)
    return


def write_global_motl_from_overlapping_subtomograms(subtomograms_path: str,
                                                    motive_list_output_dir: str,
                                                    overlap: int,
                                                    label_name: str,
                                                    output_shape: tuple,
                                                    subtomo_shape: tuple,
                                                    numb_peaks: int,
                                                    min_peak_distance: int,
                                                    number_peaks_uniquify: int,
                                                    z_shift: int) -> str:
    with h5py.File(subtomograms_path, 'r') as h5file:
        subtomos_internal_path = join(
            h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS, label_name)
        print(list(h5file[subtomos_internal_path]))
        list_of_maxima = []
        list_of_maxima_coords = []
        overlap_shift = overlap * np.array([1, 1, 1])
        z_shift_vector = [z_shift, 0, 0]
        for subtomo_name in list(h5file[subtomos_internal_path]):
            subtomo_list_of_maxima, subtomo_maxima_coords = \
                get_peaks_per_subtomo_with_overlap(
                    h5file=h5file,
                    subtomo_name=subtomo_name,
                    subtomo_shape=subtomo_shape,
                    output_shape=output_shape,
                    subtomos_internal_path=subtomos_internal_path,
                    numb_peaks=numb_peaks,
                    min_peak_distance=min_peak_distance,
                    overlap=overlap)
            print("Peaks in ", subtomo_name, " computed")
            subtomo_corner, _ = get_subtomo_corner_and_side_lengths(
                subtomo_name,
                subtomo_shape,
                output_shape)

            subtomo_maxima_coords = shift_coordinates_by_vector(
                coordinates=subtomo_maxima_coords, shift_vector=-overlap_shift)
            subtomo_maxima_coords = shift_coordinates_by_vector(
                coordinates=subtomo_maxima_coords, shift_vector=z_shift_vector)

            list_of_maxima += subtomo_list_of_maxima
            list_of_maxima_coords += subtomo_maxima_coords

        motl_file_name = unique_coordinates_motl_writer(
            path_to_output_folder=motive_list_output_dir,
            list_of_peak_scores=list_of_maxima,
            list_of_peak_coords=list_of_maxima_coords,
            number_peaks_to_uniquify=number_peaks_uniquify,
            minimum_peaks_distance=min_peak_distance)
    return motl_file_name


def write_global_motl_from_overlapping_subtomograms_multiclass(
        subtomograms_path: str,
        motive_list_output_dir: str,
        overlap: int,
        label_name: str,
        output_shape: tuple,
        subtomo_shape: tuple,
        numb_peaks: int,
        min_peak_distance: int,
        class_number: int,
        number_peaks_uniquify: int,
        z_shift: int) -> str:
    with h5py.File(subtomograms_path, 'r') as h5file:
        subtomos_internal_path = join(
            h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS, label_name)
        list_of_maxima = []
        list_of_maxima_coords = []
        overlap_shift = overlap * np.array([1, 1, 1])
        z_shift_vector = [z_shift, 0, 0]
        print(z_shift_vector)
        for subtomo_name in list(h5file[subtomos_internal_path]):
            subtomo_list_of_maxima, subtomo_maxima_coords = \
                get_peaks_per_subtomo_with_overlap_multiclass(
                    h5file=h5file,
                    subtomo_name=subtomo_name,
                    subtomo_shape=subtomo_shape,
                    output_shape=output_shape,
                    subtomos_internal_path=subtomos_internal_path,
                    numb_peaks=numb_peaks,
                    class_number=class_number,
                    min_peak_distance=min_peak_distance,
                    overlap=overlap)
            print(len(subtomo_maxima_coords), "peaks in ", subtomo_name,
                  " computed")
            subtomo_corner, _ = get_subtomo_corner_and_side_lengths(
                subtomo_name,
                subtomo_shape,
                output_shape)

            subtomo_maxima_coords = shift_coordinates_by_vector(
                coordinates=subtomo_maxima_coords, shift_vector=-overlap_shift)
            subtomo_maxima_coords = shift_coordinates_by_vector(
                coordinates=subtomo_maxima_coords, shift_vector=z_shift_vector)

            list_of_maxima += subtomo_list_of_maxima
            list_of_maxima_coords += subtomo_maxima_coords
        print("Before saving, the total number of peaks is:",
              len(list_of_maxima_coords))

        # To arrange by score is necesary, otherwise most are trash
        # from the last tomogram:
        values = []
        coordinates = []
        for val, zyx_coord in sorted(
                list(zip(list_of_maxima, list_of_maxima_coords)),
                key=lambda x: x[0], reverse=1)[:number_peaks_uniquify]:
            values += [val]
            coordinates += [zyx_coord]

        motl_file_name = unique_coordinates_motl_writer(
            path_to_output_folder=motive_list_output_dir,
            list_of_peak_scores=list_of_maxima,
            list_of_peak_coords=list_of_maxima_coords,
            number_peaks_to_uniquify=number_peaks_uniquify,
            minimum_peaks_distance=min_peak_distance,
            class_number=class_number)
    return motl_file_name


def unique_coordinates_motl_writer(path_to_output_folder: str,
                                   list_of_peak_scores: list,
                                   list_of_peak_coords: list,
                                   number_peaks_to_uniquify: int,
                                   minimum_peaks_distance: int,
                                   class_number=0,
                                   in_tom_format=False,
                                   motl_name=None,
                                   uniquify_by_score=False
                                   ) -> str:
    """
    Motl writer for given coordinates and score values. The format of resuting
    motl follows the TOM package one: 20 columns and N rows, where N is the
    number of coordinates to be stored.
    This function uniquifies the coordinates for a given minimum distance
    between peaks.

    :param path_to_output_folder: Destination folder
    :param list_of_peak_scores: list of scores associated to each coordinate
    :param list_of_peak_coords: list of coordinates
    :param number_peaks_to_uniquify: number of coordinates to filter for
    unique coordinates. Useful when N is huge.
    :param minimum_peaks_distance:  minimum distance between peaks to be
    considered as different particles.
    :param class_number: parameter that can be used for the motl name
    :param in_tom_format: True if coordinates are in x, y, z format according to
    TOM reader.
    :param motl_name: default is None, otherwise it can be an optional string
    with the name of output motl file.
    :param uniquify_by_score: if True, when filtering to uniquify we would keep
    the repeated coordinate holding the highest score value. By default is
    set to False.
    :return: motl file path
    """

    # Arrange by score value (necessary step to not get the trash from low
    # scores when analyzing peaks from cnn):
    values = []
    coordinates = []
    # To arrange by score:
    for val, zyx_coord in sorted(
            list(zip(list_of_peak_scores, list_of_peak_coords)),
            key=lambda x: x[0], reverse=1):
        values += [val]
        coordinates += [zyx_coord]

    start = time.time()
    values, coordinates = filtering_duplicate_coords_with_values(
        motl_coords=coordinates[:number_peaks_to_uniquify],
        motl_values=values[:number_peaks_to_uniquify],
        min_peak_distance=minimum_peaks_distance,
        preference_by_score=uniquify_by_score)
    end = time.time()
    print("elapsed time for filtering coordinates", end - start, "sec")
    numb_peaks = len(values)

    if motl_name is None:
        motl_name = 'motl_' + str(numb_peaks) + '_class_' + str(
            class_number) + '.csv'
    else:
        print("motif list name given as ", motl_name)

    motl_file_name = motl_writer(
        path_to_output_folder=path_to_output_folder,
        list_of_peak_scores=values,
        list_of_peak_coords=coordinates,
        in_tom_format=in_tom_format,
        order_by_score=False,
        motl_name=motl_name)
    return motl_file_name


def write_union_of_motls(path_to_motl_1: str, path_to_motl_2: str,
                         path_to_output_folder: str,
                         min_peak_distance: int,
                         in_tom_format=False,
                         motl_name=None,
                         ) -> str:
    values, coordinates = union_of_motls(path_to_motl_1=path_to_motl_1,
                                         path_to_motl_2=path_to_motl_2)
    number_peaks_uniquify = len(values)
    motl_file_name = unique_coordinates_motl_writer(
        path_to_output_folder=path_to_output_folder,
        list_of_peak_scores=values,
        list_of_peak_coords=coordinates,
        number_peaks_to_uniquify=number_peaks_uniquify,
        minimum_peaks_distance=min_peak_distance,
        in_tom_format=in_tom_format,
        motl_name=motl_name,
        uniquify_by_score=False)
    return motl_file_name
