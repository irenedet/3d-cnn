from typing import Tuple
from os.path import join
import h5py

import numpy as np
import skimage.morphology as morph

from coordinates_toolbox.subtomos import \
    get_subtomo_corner_side_lengths_and_zero_padding
from naming import h5_internal_paths
from naming.particles import create_particle_file_name


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
    return np.array(motl[:, 7:10])


def extract_coordinates_and_values_from_em_motl(motl: np.array) -> np.array:
    values = np.array(motl[:, 0])
    coordinates = np.array(motl[:, 7:10])
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
                                           preference_by_score=True,
                                           max_num_points: float = np.inf):
    """
    Filter out coordinates that are close by a given radius to each other.
    :param motl_coords: list of coordinate points to filter
    :param motl_values: list of score values associated
    :param min_peak_distance: radius of tolerance for distance between points
    to filter
    :param preference_by_score: Boolean, when True, the peak of highest score
    will be kept and the neighbouring coordinates with lower score will be
    filtered out.
    :param max_num_points: optional, if a maximum number of coordinates should
    be considered
    :return: a list of coordinate points where none of them are close to the
    rest by the given radius
    """
    motl_coords = np.array(motl_coords)
    unique_motl_coords = [motl_coords[0]]
    unique_motl_values = [motl_values[0]]

    for value, point in zip(motl_values[1:], motl_coords[1:]):
        flag = "unique"
        n_point = 0
        n_unique_points = len(unique_motl_coords)
        if n_unique_points < max_num_points:
            while flag == "unique" and n_point < n_unique_points:
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


def average_duplicated_centroids(motl_coords: list, cluster_size_list: list,
                                 min_peak_distance: int):
    unique_motl_coords = [motl_coords[0]]
    unique_cluster_size_list = [cluster_size_list[0]]
    for point, weight_point in zip(motl_coords[1:], cluster_size_list[1:]):
        flag = "unique"
        n_point = 0
        while flag == "unique" and n_point < len(unique_motl_coords):
            x = unique_motl_coords[n_point]
            weight_x = unique_cluster_size_list[n_point]
            n_point += 1
            if np.linalg.norm(x - point) <= min_peak_distance:
                flag = "repeated"
                total_weight = weight_point + weight_x
                rel_weight_x = weight_x / total_weight
                rel_weight_point = weight_point / total_weight
                relative_mean = np.array(x) * rel_weight_x + \
                                np.array(point) * rel_weight_point
                print("relative_mean", relative_mean)
                unique_motl_coords[n_point - 1] = [int(coord) for coord in
                                                   relative_mean]
                unique_cluster_size_list[n_point - 1] = np.mean(
                    [weight_point, weight_x])
        if flag == "unique":
            unique_motl_coords += [point]
            unique_cluster_size_list += [weight_point]
    return unique_motl_coords, unique_cluster_size_list


def get_clusters_within_size_range(dataset: np.array, min_cluster_size: int,
                                   max_cluster_size: int, connectivity=1):
    assert min_cluster_size < max_cluster_size

    labeled_clusters, num = morph.label(input=dataset, background=0,
                                        return_num=True,
                                        connectivity=connectivity)
    labels_list, cluster_size = np.unique(labeled_clusters, return_counts=True)
    labels_list_within_range = labels_list[(cluster_size > min_cluster_size) & (
            cluster_size < max_cluster_size)]
    cluster_size_within_range = list(
        cluster_size[(cluster_size > min_cluster_size) & (
                cluster_size < max_cluster_size)])
    print("Clusters in subtomo before size filtering =", num)
    return labeled_clusters, labels_list_within_range, cluster_size_within_range


def get_cluster_centroids(dataset: np.array, min_cluster_size: int,
                          max_cluster_size: int, connectivity=1) -> tuple:
    labeled_clusters, labels_list_within_range, cluster_size_within_range = \
        get_clusters_within_size_range(dataset=dataset,
                                       min_cluster_size=min_cluster_size,
                                       max_cluster_size=max_cluster_size,
                                       connectivity=connectivity)
    centroids_list = list()
    for label in labels_list_within_range:
        cluster = np.where(labeled_clusters == label)
        centroid = np.rint(np.mean(cluster, axis=1))
        centroids_list.append(centroid)
    print("Clusters in subtomo after size filtering =", len(centroids_list))
    return labeled_clusters, centroids_list, cluster_size_within_range


def get_cluster_centroids_from_partition(partition: str, label_name: str,
                                         min_cluster_size: int,
                                         max_cluster_size: int,
                                         output_shape: tuple,
                                         segmentation_class=0,
                                         overlap=12) -> tuple:
    with h5py.File(partition, "a") as f:
        internal_path = join(
            h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS,
            label_name)
        labels_path = join(h5_internal_paths.CLUSTERING_LABELS, label_name)
        subtomo_names = list(f[internal_path])
        print(len(subtomo_names), " subtomos in this partition.")
        full_centroids_list = list()
        full_cluster_size_list = list()
        for subtomo_name in list(subtomo_names):
            print("subtomo_name", subtomo_name)
            subtomo_path = join(internal_path, subtomo_name)
            subtomo_data = f[subtomo_path][segmentation_class, ...]
            subtomo_shape = subtomo_data.shape
            # Extract subtomo minus overlap

            # extract the subtomo data in the internal subtomo plus a bit more
            # (overlap//2), instead of extracting sharply
            subtomo_corner, subtomo_side_lengths, zero_border_thickness = \
                get_subtomo_corner_side_lengths_and_zero_padding(subtomo_name,
                                                                 subtomo_shape,
                                                                 output_shape,
                                                                 overlap // 2)

            shape_minus_overlap = tuple([dim - pad[0] - pad[1] for pad, dim in
                                         zip(zero_border_thickness,
                                             subtomo_data.shape)])

            mask_out_half_overlap = np.ones(shape_minus_overlap)
            mask_out_half_overlap = np.pad(mask_out_half_overlap,
                                           zero_border_thickness, "constant")

            subtomo_data = mask_out_half_overlap * subtomo_data

            # Threshold segmentation
            subtomo_data = 1 * (subtomo_data == 1)
            # Get centroids per subtomo

            subtomo_labels, subtomo_centroids_list, cluster_size_list = \
                get_cluster_centroids(dataset=subtomo_data,
                                      min_cluster_size=min_cluster_size,
                                      max_cluster_size=max_cluster_size)

            if subtomo_name not in list(f[labels_path]):
                subtomo_labels_path = join(labels_path, subtomo_name)
                f[subtomo_labels_path] = subtomo_labels[:]
            else:
                print("Clustering label already exists.")

            overlap_shift = np.array([overlap, overlap, overlap])
            shift_vector = np.array(subtomo_corner) - overlap_shift

            # Shift subtomo coordinates to global coordinate system
            subtomo_centroids_list = \
                shift_coordinates_by_vector(subtomo_centroids_list,
                                            shift_vector)

            full_centroids_list += subtomo_centroids_list
            full_cluster_size_list += cluster_size_list
    return full_centroids_list, full_cluster_size_list


def get_cluster_centroids_from_full_dataset(partition: str, label_name: str,
                                            min_cluster_size: int,
                                            max_cluster_size: int,
                                            output_shape: tuple,
                                            segmentation_class=0,
                                            overlap=12) -> list:
    with h5py.File(partition, "a") as f:
        internal_path = join(
            h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS,
            label_name)
        labels_path = join(h5_internal_paths.CLUSTERING_LABELS, label_name)
        subtomo_names = list(f[internal_path])
        print(len(subtomo_names), " subtomos in this partition.")
        full_centroids_list = list()
        for subtomo_name in list(subtomo_names):
            print("subtomo_name", subtomo_name)
            subtomo_path = join(internal_path, subtomo_name)
            subtomo_data = f[subtomo_path][segmentation_class, ...]
            subtomo_shape = subtomo_data.shape
            # Extract subtomo minus overlap

            # extract the subtomo data in the internal subtomo plus a bit more
            # (overlap//2), instead of extracting sharply
            subtomo_corner, subtomo_side_lengths, zero_border_thickness = \
                get_subtomo_corner_side_lengths_and_zero_padding(subtomo_name,
                                                                 subtomo_shape,
                                                                 output_shape,
                                                                 overlap // 2)

            shape_minus_overlap = tuple([dim - pad[0] - pad[1] for pad, dim in
                                         zip(zero_border_thickness,
                                             subtomo_data.shape)])

            mask_out_half_overlap = np.ones(shape_minus_overlap)
            mask_out_half_overlap = np.pad(mask_out_half_overlap,
                                           zero_border_thickness, "constant")

            subtomo_data = mask_out_half_overlap * subtomo_data

            # Threshold segmentation
            subtomo_data = 1 * (subtomo_data == 1)
            # Get centroids per subtomo

            subtomo_labels, subtomo_centroids_list = \
                get_cluster_centroids(dataset=subtomo_data,
                                      min_cluster_size=min_cluster_size,
                                      max_cluster_size=max_cluster_size)

            if subtomo_name not in list(f[labels_path]):
                subtomo_labels_path = join(labels_path, subtomo_name)
                f[subtomo_labels_path] = subtomo_labels[:]
            else:
                print("Clustering label already exists.")

            overlap_shift = np.array([overlap, overlap, overlap])
            shift_vector = np.array(subtomo_corner) - overlap_shift

            # Shift subtomo coordinates to global coordinate system
            subtomo_centroids_list = \
                shift_coordinates_by_vector(subtomo_centroids_list,
                                            shift_vector)

            full_centroids_list += subtomo_centroids_list
    return full_centroids_list


def shift_coordinates_by_vector(coordinates: list,
                                shift_vector: np.array) -> list:
    return [np.array(coord) + np.array(shift_vector) for coord in
            np.array(coordinates)]
