import csv
import datetime
import re
import time
from os.path import join

import h5py
import numpy as np

from src.python.coordinates_toolbox.utils import \
    filtering_duplicate_coords_with_values
from src.python.naming import h5_internal_paths
from src.python.peak_toolbox.subtomos import \
    _get_peaks_per_subtomo_with_overlap, \
    _get_peaks_per_subtomo_with_overlap_multiclass, \
    _get_subtomo_corner_and_side_lengths, _shift_coordinates_by_vector

from src.python.coordinates_toolbox.utils import rearrange_hdf_coordinates


def motl_writer(path_to_output_folder: str, list_of_peak_scores: list,
                list_of_peak_coords: list, in_tom_format=False):
    """
    Already modified to match em_motl format
    """

    numb_peaks = len(list_of_peak_scores)
    motl_file_name = join(path_to_output_folder,
                          'motl_' + str(numb_peaks) + '.csv')
    with open(motl_file_name, 'w', newline='') as csvfile:
        motlwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        indx = 0
        for val, point in sorted(
                list(zip(list_of_peak_scores, list_of_peak_coords)),
                key=lambda x: x[0], reverse=1):
            indx += 1
            if in_tom_format:
                x, y, z = point
            else:
                x, y, z = rearrange_hdf_coordinates(point)
            motlwriter.writerow([str(val) + ',0,0,' + str(
                indx) + ',0,0,0,' + str(x) + ',' + str(y) + ',' + str(
                z) + ',0,0,0,0,0,0,0,0,0,1'])
    print("The motive list has been writen in", motl_file_name)
    return


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
            _, _, _, auPRC, _ = jobs_statistics_dict[job_name]
            row = [job_name, k, param_value, classes, auPRC]
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
                                                    z_shift: int
                                                    ) -> str:
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
                _get_peaks_per_subtomo_with_overlap(
                    h5file=h5file,
                    subtomo_name=subtomo_name,
                    subtomo_shape=subtomo_shape,
                    output_shape=output_shape,
                    subtomos_internal_path=subtomos_internal_path,
                    numb_peaks=numb_peaks,
                    min_peak_distance=min_peak_distance,
                    overlap=overlap)
            print("Peaks in ", subtomo_name, " computed")
            subtomo_corner, _ = _get_subtomo_corner_and_side_lengths(
                subtomo_name,
                subtomo_shape,
                output_shape)

            subtomo_maxima_coords = _shift_coordinates_by_vector(
                coordinates=subtomo_maxima_coords, shift_vector=-overlap_shift)
            subtomo_maxima_coords = _shift_coordinates_by_vector(
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
        print(list(h5file[subtomos_internal_path]))
        list_of_maxima = []
        list_of_maxima_coords = []
        overlap_shift = overlap * np.array([1, 1, 1])
        z_shift_vector = [z_shift, 0, 0]
        for subtomo_name in list(h5file[subtomos_internal_path]):
            subtomo_list_of_maxima, subtomo_maxima_coords = \
                _get_peaks_per_subtomo_with_overlap_multiclass(
                    h5file=h5file,
                    subtomo_name=subtomo_name,
                    subtomo_shape=subtomo_shape,
                    output_shape=output_shape,
                    subtomos_internal_path=subtomos_internal_path,
                    numb_peaks=numb_peaks,
                    class_number=class_number,
                    min_peak_distance=min_peak_distance,
                    overlap=overlap)
            print("Peaks in ", subtomo_name, " computed")
            subtomo_corner, _ = _get_subtomo_corner_and_side_lengths(
                subtomo_name,
                subtomo_shape,
                output_shape)

            subtomo_maxima_coords = _shift_coordinates_by_vector(
                coordinates=subtomo_maxima_coords, shift_vector=-overlap_shift)
            subtomo_maxima_coords = _shift_coordinates_by_vector(
                coordinates=subtomo_maxima_coords, shift_vector=z_shift_vector)

            list_of_maxima += subtomo_list_of_maxima
            list_of_maxima_coords += subtomo_maxima_coords

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
                                   class_number=0) -> str:
    """
    Already modified to match em_motl format
    """
    values = []
    coordinates = []
    for val, zyx_coord in sorted(
            list(zip(list_of_peak_scores, list_of_peak_coords)),
            key=lambda x: x[0], reverse=1):
        values += [val]
        coordinates += [zyx_coord]

    start = time.time()
    values, coordinates = filtering_duplicate_coords_with_values(
        motl_coords=coordinates[:number_peaks_to_uniquify],
        motl_values=values[:number_peaks_to_uniquify],
        min_peak_distance=minimum_peaks_distance)
    end = time.time()
    print("elapsed time for filtering coordinates", end - start, "sec")
    numb_peaks = len(values)
    motl_file_name = join(path_to_output_folder,
                          'motl_' + str(numb_peaks) + '_class_' + str(
                              class_number) + '.csv')
    with open(motl_file_name, 'w', newline='') as csvfile:
        motlwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        indx = 0

        for val, zyx_coord in zip(values, coordinates):
            indx += 1
            x, y, z = rearrange_hdf_coordinates(zyx_coord)
            motlwriter.writerow([str(val) + ',0,0,' + str(
                indx) + ',0,0,0,' + str(x) + ',' + str(y) + ',' + str(
                z) + ',0,0,0,0,0,0,0,0,0,1'])
        print("The motive list has been writen in", motl_file_name)
    return motl_file_name
