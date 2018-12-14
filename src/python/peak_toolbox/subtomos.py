import numpy as np
import h5py
from os.path import join
from functools import reduce
from src.python.coordinates_toolbox import subtomos
from src.python.naming import h5_internal_paths
from src.python.peak_toolbox.utils import extract_peaks


def _get_numb_peaks(subtomo_shape: tuple, min_peak_distance: int) -> int:
    numb_peaks = [shape / 2 / min_peak_distance for shape in
                  subtomo_shape]
    numb_peaks = reduce(lambda x, y: x * y, numb_peaks)
    return int(numb_peaks)


def _shift_subtomo_maxima_coords(subtomo_maxima_coords: list,
                                 init_points: list) -> list:
    return [coord + np.array(init_points) for coord in
            np.array(subtomo_maxima_coords)]


def _extract_data_subtomo(h5file: h5py._hl.files.File,
                          subtomo_h5_internal_path: str,
                          subtomo_side_lengths: list) -> np.array:
    return h5file[subtomo_h5_internal_path][0, :subtomo_side_lengths[0],
           :subtomo_side_lengths[1], :subtomo_side_lengths[2]]


def _get_subtomo_corner_and_sidelengths(subtomo_name: str, subtomo_shape: tuple,
                                        output_shape: tuple) -> tuple:
    subtomo_center = subtomos.get_coord_from_name(subtomo_name)
    init_points, _, subtomo_side_lengths = \
        subtomos.get_subtomo_corners(output_shape, subtomo_shape,
                                     subtomo_center)
    return init_points, subtomo_side_lengths


def _get_peaks_per_subtomo(h5file: h5py._hl.files.File, subtomo_name: str,
                           subtomo_shape: tuple, output_shape: tuple,
                           subtomos_internal_path: str, numb_peaks: int,
                           min_peak_distance: int) -> tuple:
    subtomo_corner, subtomo_side_lengths = \
        _get_subtomo_corner_and_sidelengths(subtomo_name,
                                            subtomo_shape,
                                            output_shape)
    print("Subtomogram corner", subtomo_corner)
    subtomo_h5_internal_path = join(subtomos_internal_path,
                                    subtomo_name)

    data_subtomo = _extract_data_subtomo(h5file, subtomo_h5_internal_path,
                                         subtomo_side_lengths)

    subtomo_list_of_maxima, subtomo_list_of_maxima_coords = \
        extract_peaks(dataset=data_subtomo, numb_peaks=numb_peaks,
                      radius=min_peak_distance)

    shifted_subtomo_maxima_coords = \
        _shift_subtomo_maxima_coords(subtomo_list_of_maxima_coords,
                                     subtomo_corner)
    return subtomo_list_of_maxima, shifted_subtomo_maxima_coords


def get_peaks_from_subtomograms(subtomo_file_path: str, label_name: str,
                                subtomo_shape: tuple,
                                output_shape: tuple,
                                min_peak_distance: int) -> tuple:
    list_of_maxima = []
    list_of_maxima_coords = []
    numb_peaks = _get_numb_peaks(subtomo_shape, min_peak_distance)
    print("Number of peaks per subtomogram will be", numb_peaks)
    with h5py.File(subtomo_file_path, 'r') as h5file:
        subtomos_internal_path = join(
            h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS, label_name)
        for subtomo_name in list(h5file[subtomos_internal_path]):
            subtomo_list_of_maxima, shifted_subtomo_maxima_coords = \
                _get_peaks_per_subtomo(h5file, subtomo_name, subtomo_shape,
                                       output_shape,
                                       subtomos_internal_path, numb_peaks,
                                       min_peak_distance)

            list_of_maxima += subtomo_list_of_maxima
            list_of_maxima_coords += shifted_subtomo_maxima_coords
    return list_of_maxima, list_of_maxima_coords
