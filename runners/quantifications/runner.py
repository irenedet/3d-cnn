import os
import random
from time import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.signal import convolve, correlate
from scipy.spatial.distance import cdist
from tqdm import tqdm

# from constants.dataset_tables import ModelsTableHeader, DatasetTableHeader
# from file_actions.readers.tomograms import load_tomogram
from file_actions.writers.csv import build_tom_motive_list
from constants.dataset_tables import DatasetTableHeader
from file_actions.readers.tomograms import load_tomogram
from tomogram_utils.peak_toolbox.utils import _generate_unit_particle
from tomogram_utils.peak_toolbox.utils import read_motl_coordinates_and_values


def get_minimum_distance_between_motls(coordinates1: list, coordinates2: list = None):
    if coordinates2 is None:
        d = cdist(coordinates1, coordinates1)
        d_mod = d + np.max(d) * np.eye(d.shape[0])
        min_distance = np.min(d_mod)
    else:
        coordinates1, coordinates2 = np.array(coordinates1), np.array(coordinates2)
        d = cdist(coordinates1, coordinates2)
        min_distance = np.min(d)
    return min_distance


def get_minimum_distance_distribution_between_motls(coordinates1: list, coordinates2: list = None,
                                                    voxel_size: float = 1):
    if coordinates2 is None:
        coordinates1 = np.array(coordinates1)
        d = cdist(coordinates1, coordinates1)
        d_mod = d + np.max(d) * np.eye(d.shape[0])
        min_distance_distribution = d_mod.min(axis=1)
    else:
        coordinates1, coordinates2 = np.array(coordinates1), np.array(coordinates2)
        d = cdist(coordinates1, coordinates2)
        min_distance_distribution = d.min(axis=1)
    return voxel_size * min_distance_distribution


def get_mask_volume(mask: np.array, volume_per_voxel: float, mask_level: float = 1):
    segmentation_points = np.transpose(np.array(np.where(mask == mask_level)))
    total_voxels_in_mask = len(segmentation_points)
    total_volume = volume_per_voxel * total_voxels_in_mask
    return total_volume


def get_distance_to_mask(coordinates: list, mask: np.array, value: float = 1, voxel_size: float = 1):
    coordinates = np.array(coordinates)
    segmentation_points = np.transpose(np.array(np.where(mask == value)))
    distance_matrix = cdist(coordinates, segmentation_points, 'euclidean')
    min_distance_distribution = distance_matrix.min(axis=1)
    return voxel_size * min_distance_distribution


def get_distance_motl_to_mask(dataset_table, tomo_name, particle, mask, voxel_size):
    df = pd.read_csv(dataset_table)
    DTHeader_masks = DatasetTableHeader(semantic_classes=[mask])
    DTHeader_particles = DatasetTableHeader(semantic_classes=[particle])

    tomo_row = df.loc[df[DTHeader_masks.tomo_name] == tomo_name]

    mask_column = DTHeader_masks.masks_names[0]
    mask_path = tomo_row[mask_column].values[0]

    motl_column = DTHeader_particles.clean_motls[0]
    motl_path = tomo_row[motl_column].values[0]

    _, motl_coords = read_motl_coordinates_and_values(path_to_motl=motl_path)
    segmentation_mask = load_tomogram(path_to_dataset=mask_path)
    # TODO: Check this way to transpose!!
    transposed_coords = [[p[2], p[1], p[0]] for p in motl_coords]
    print("starting to calculate distance")
    start = time()
    distance_distribution = get_distance_to_mask(coordinates=transposed_coords, mask=segmentation_mask, value=1,
                                                 voxel_size=voxel_size)
    end = time()
    print("elapsed time: {} secs.".format(end - start))
    return distance_distribution


def write_quantification_value(statistics_file: str, statistics_label: str, tomo_name: str, stat_measure: float):
    dict_stats = {'tomo_name': [tomo_name],
                  statistics_label: [stat_measure],
                  }
    mini_stats_df = pd.DataFrame(dict_stats)
    if os.path.isfile(statistics_file):
        print("The statistics file exists")
        stats_df = pd.read_csv(statistics_file)
        stats_df['tomo_name'] = stats_df['tomo_name'].astype(str)
        if statistics_label in stats_df.keys():
            print("Model's statistics label exists")
            if tomo_name in stats_df['tomo_name'].values:
                row = stats_df['tomo_name'] == tomo_name
                stats_df.loc[row, statistics_label] = [stat_measure]
            else:
                stats_df = stats_df.append(mini_stats_df, sort=False)
        else:
            print("Tomo name does not exist")
            stats_df = pd.merge(stats_df, mini_stats_df, on='tomo_name',
                                how='outer')
        stats_df.to_csv(path_or_buf=statistics_file, index=False)

    else:
        print("The statistics file does not exist, we will create it.")
        path = os.path.dirname(statistics_file)
        os.makedirs(path, exist_ok=True)
        mini_stats_df.to_csv(path_or_buf=statistics_file, index=False)
    return


def write_distribution(statistics_file: str, tomo_name: str, stat_measure: float):
    dict_stats = {tomo_name: stat_measure}
    mini_stats_df = pd.DataFrame(dict_stats)
    if os.path.isfile(statistics_file):
        print("The statistics file exists")
        stats_df = pd.read_csv(statistics_file)
        if tomo_name in stats_df.keys():
            print("Tomo's statistics label exists")
        else:
            stats_df = pd.concat([stats_df, mini_stats_df], axis=1, sort=False)
            stats_df.to_csv(path_or_buf=statistics_file, index=False)
    else:
        print("The statistics file does not exist, we will create it.")
        path = os.path.dirname(statistics_file)
        os.makedirs(path, exist_ok=True)
        mini_stats_df.to_csv(path_or_buf=statistics_file, index=False)
    return


def extract_motl(particle: str, dataset_table: str, tomo_name: str):
    df = pd.read_csv(dataset_table)
    DTHeader_particles = DatasetTableHeader(semantic_classes=[particle])
    motl_column = DTHeader_particles.clean_motls
    tomo_row = df.loc[df[DTHeader_particles.tomo_name] == tomo_name]
    motl_path = tomo_row[motl_column].values[0][0]
    return motl_path


def get_points_in_mask(coordinates: list, mask: np.array, value: float = 1):
    segmentation_points = list(np.transpose(np.array(np.where(mask == value), dtype=int)))
    coordinates = [tuple(point) for point in coordinates]
    segmentation_points = [tuple(point) for point in segmentation_points]
    print("len(coordinates) =", len(coordinates))
    print("len(mask_voxels_count) =", len(segmentation_points))
    print("sample of coords: ", coordinates[:5])
    print("sample of voxels in mask: ", segmentation_points[:5])

    count_coordinates_in_mask = len(set(segmentation_points) & set(coordinates))
    print("Number of particles in mask:", count_coordinates_in_mask)
    return count_coordinates_in_mask


def get_points_density_within_mask(dataset_table, tomo_name, particle, mask, voxel_size):
    df = pd.read_csv(dataset_table)
    DTHeader_masks = DatasetTableHeader(semantic_classes=[mask])
    DTHeader_particles = DatasetTableHeader(semantic_classes=[particle])

    tomo_row = df.loc[df[DTHeader_masks.tomo_name] == tomo_name]

    mask_column = DTHeader_masks.masks_names[0]
    mask_path = tomo_row[mask_column].values[0]
    print(mask_path)

    motl_column = DTHeader_particles.clean_motls[0]
    motl_path = tomo_row[motl_column].values[0]

    _, motl_coords = read_motl_coordinates_and_values(path_to_motl=motl_path)
    segmentation_mask = load_tomogram(path_to_dataset=mask_path)

    transposed_coords = [[p[2], p[1], p[0]] for p in motl_coords]
    print("starting to calculate mask volume")
    start = time()
    volume_per_voxel = voxel_size ** 3
    mask_volume = get_mask_volume(mask=segmentation_mask, volume_per_voxel=volume_per_voxel, mask_level=1)
    end = time()
    print("elapsed time: {} secs.".format(end - start))

    print("starting to count particles within mask")
    start = time()
    count_points_in_mask = get_points_in_mask(coordinates=transposed_coords, mask=segmentation_mask, value=1)
    end = time()
    print("elapsed time: {} secs.".format(end - start))
    return count_points_in_mask / mask_volume


def get_points_in_mask(coordinates: list, segmentation_points: list):
    coordinates = [tuple(point) for point in coordinates]
    segmentation_points = [tuple(point) for point in segmentation_points]
    print("len(coordinates) =", len(coordinates))
    print("len(mask_voxels_count) =", len(segmentation_points))
    coordinates_in_mask = set(segmentation_points) & set(coordinates)
    coordinates_in_mask = [list(point) for point in coordinates_in_mask]
    return coordinates_in_mask


def generate_ball_mask(shape: tuple, center: tuple or list, radius: int):
    a, b, c = center
    nx, ny, nz = shape
    y, x, z = np.ogrid[-a:nx - a, -b:ny - b, -c:nz - c]
    ball = x * x + y * y + z * z < radius * radius
    return ball


def generate_ball_mask2(shape: tuple, center: tuple or list, radius: int):
    ball = np.zeros(shape, dtype=int)
    ball_coords = _generate_unit_particle(radius=radius)
    shifted_ball = np.array(ball_coords) + np.array(center)
    ball[tuple(shifted_ball.T)] = 1
    return ball


def calculate_ball_intersection_ratio(domain: np.array, ball_mask: np.array):
    return np.sum(domain * ball_mask) / np.sum(ball_mask)


def generate_Eest(coords: np.array, r_min: float, r_max: float, num: int = 1):
    radius_range = np.linspace(start=r_min, stop=r_max, num=num)
    E_est = []
    # start = time()
    d = list(cdist(coords, coords))
    for r in radius_range:
        V_r = (4 / 3) * np.pi * r ** 3
        N_points = []
        for center, d_row in zip(coords, d):
            points_within_radius = len(np.where(d_row < r)[0])
            N_points.append(points_within_radius - 1)
        E_r = np.mean(N_points)
        E_est.append(E_r / V_r)
    # end = time()
    # print("elapsed time: {} secs.".format(end - start))
    return E_est, radius_range


def get_wij(A, xi, r):
    """
    Needs to be optimized....
    :param A:
    :param xi:
    :param r:
    :return:
    """
    nx, ny, nz = A.shape
    a, b, c = xi
    y, x, z = np.ogrid[-a:nx - a, -b:ny - b, -c:nz - c]
    ball = (x * x + y * y + z * z <= r * r)
    return np.sum(ball * A)


def get_wij2(A, xi, r):
    shape = A.shape
    ball = np.zeros(shape, dtype=int)
    center = xi
    ball_coords = _generate_unit_particle(radius=r)
    shifted_ball = np.array(ball_coords) + np.array(center)
    ball[tuple(shifted_ball.T)] = 1
    return np.sum(ball * A)


def get_centered_grid(shape: tuple, center: tuple or list or np.array):
    nx, ny, nz = shape
    a, b, c = center
    y, x, z = np.ogrid[-a:nx - a, -b:ny - b, -c:nz - c]
    return x, y, z


def get_tight_ball_mask(radius: float) -> np.array:
    shape = 2 * radius + 1, 2 * radius + 1, 2 * radius + 1
    center = radius, radius, radius
    y, x, z = get_centered_grid(shape=shape, center=center)
    ball = 1 * (x * x + y * y + z * z <= radius * radius)
    return ball


def get_wij3(A: np.array, xi: tuple or list or np.array, radius: float) -> float:
    ball = get_tight_ball_mask(radius=radius)
    y, x, z = get_centered_grid(shape=ball.shape, center=xi)
    A_submatrix = A[x, y, z]
    A_conv_ball = convolve(A_submatrix, ball, mode='valid', method='direct')
    wij = A_conv_ball[0, 0, 0]
    return wij


def get_grid_around_center(shape: tuple, center: tuple or list or np.array):
    """
    Creates a grid of indexes in the region [a-dx-1 : a+dx, b-dy-1:b+dy, c-dz-1:c+dz]
    where dx, dy, dz = nx//2, ny//2, nz//2.
    :param shape:
    :param center:
    :return:
    """
    nx, ny, nz = shape
    dx, dy, dz = nx // 2, ny // 2, nz // 2
    a, b, c = center
    y, x, z = np.ogrid[a - dx - 1:dx + a, b - dy - 1:dy + b, c - dz - 1:dz + c]
    return x, y, z


def get_wij4(domain: np.array, xi: tuple or list or np.array, radius: float) -> float:
    ball = get_tight_ball_mask(radius=radius)
    ball_vol = 4 * np.pi * radius ** 3 / 3
    y, x, z = get_grid_around_center(shape=ball.shape, center=xi)
    square_subdomain = domain[x, y, z]
    intersection = convolve(square_subdomain, ball, mode='valid', method='direct')
    wij = intersection[0, 0, 0] / ball_vol
    return wij


def get_Ripley_Kest(A, coords, r_min, r_max, num):
    n = len(coords)
    d = cdist(coords, coords)
    r = np.sort(d, axis=1)[:, 1:]

    w = np.zeros(r.shape)
    for i, xi in enumerate(coords):
        for j in range(n - 1):
            rij = r[i, j]
            w[i, j] = get_wij(A=A, xi=xi, r=rij)

    K = []
    radius_range = np.linspace(start=r_min, stop=r_max, num=num)
    for t in radius_range:
        K_t = 0
        for i, xi in enumerate(coords):
            ri = r[i, :]
            indicator = (ri < t)
            I_t = np.sum(indicator)
            if I_t > 0:
                wij = np.max(w[i, indicator])
                K_t += I_t / wij
        K.append(K_t)

    K = np.sum(A) * np.array(K) / n / (n - 1)

    return K, radius_range


def get_Ripley_Kest3(A, coords, r_min, r_max, num) -> Tuple:
    n = len(coords)
    d = cdist(coords, coords)
    r = np.sort(d, axis=1)[:, 1:]
    A_padded = np.pad(A, pad_width=r_max, mode='constant')  # todo check
    # print("Staring to generate edge correctors w!")
    w = np.ones(r.shape)
    for i, xi in enumerate(coords):
        xi = np.array(xi) + r_max * np.array([1, 1, 1])  # todo check
        for j in range(n - 1):
            rij = r[i, j]
            if rij <= r_max:
                # wij = get_wij3(A=A, xi=xi, radius=rij)
                wij = get_wij4(domain=A_padded, xi=xi, radius=rij)  # todo check
                if wij > 0:
                    # this in pple should not be done, it's in case the ball does not intersect the mask A == 1
                    w[i, j] = wij

    K = []
    radius_range = np.linspace(start=r_min, stop=r_max, num=num)
    for t in radius_range:
        K_t = 0
        for i, xi in enumerate(coords):
            ri = r[i, :]
            indicator = (ri <= t)
            I_t = np.sum(indicator) - 1
            if I_t > 0:
                wij = np.max(w[i, indicator])
                K_t += I_t / wij
        K.append(K_t)

    K = np.sum(A) * np.array(K) / n / (n - 1)

    return K, radius_range


def save_Kest_fig(fig_path: str, Kest: list or np.array, envelope_min: list or np.array,
                  envelope_max: list or np.array, particle: str, percent: float, voxel_size_nm: float,
                  radius_range: list or np.array):
    radius_range_nm = voxel_size_nm * np.array(radius_range)
    r_min_nm, r_max_nm = np.min(radius_range_nm), np.max(radius_range_nm)
    envelope_mean = 0.5 * (envelope_min + envelope_max)
    fig, axs = plt.subplots()
    axs.plot(voxel_size_nm * np.array(radius_range), 0 * envelope_mean, "-", color='lightgreen')
    axs.fill_between(voxel_size_nm * np.array(radius_range), envelope_min - envelope_mean,
                     envelope_max - envelope_mean, color='lightgreen', alpha=0.5)
    axs.plot(voxel_size_nm * np.array(radius_range), np.array(Kest) - envelope_mean, "--o",
             color='cornflowerblue')
    title = "K (Ripley's estimate): " + particle
    axs.set_title(title)
    axs.set_xlabel("radius (nm)")
    axs.set_xlim([r_min_nm, r_max_nm])
    axs.set_ylabel("K")
    envelope_label = "$K_{CSR} - mean(K_{CSR})$ envelope at " + str(percent) + "%"
    labels = ["$K_{est} - mean(K_{CSR})$",
              envelope_label]
    handles = [Line2D(xdata=[0], ydata=[0], color='cornflowerblue', marker='o'),
               plt.Rectangle((0, 0), 0.5, 0.1, color='lightgreen', alpha=0.5)]

    axs.legend(handles, labels, loc="upper right", shadow=True)

    plt.savefig(fig_path)
    return


"""
New version
"""


# noinspection PyArgumentList
# def get_minimum_distance_distribution_between_motls(coordinates1: list, coordinates2: list = None,
#                                                     voxel_size: float = 1):
#     if coordinates2 is None:
#         coordinates1 = np.array(coordinates1)
#         d = cdist(coordinates1, coordinates1)
#         d_mod = d + np.max(d) * np.eye(d.shape[0])
#         min_distance_distribution = d_mod.min(axis=1)
#     else:
#         coordinates1, coordinates2 = np.array(coordinates1), np.array(coordinates2)
#         d = cdist(coordinates1, coordinates2)
#         min_distance_distribution = d.min(axis=1)
#     return voxel_size * min_distance_distribution


def get_grid_around_origin(shape: tuple, center: tuple or list or np.array):
    """
    Creates a grid of indexes in the region [-dx1:dx2, -dy1:dy2, -dz1:dz2],
    where dx1, dx2 = -a, nx-a etc.
    :param shape:
    :param center:
    :return:
    """
    nx, ny, nz = shape
    a, b, c = center
    y, x, z = np.ogrid[-a:nx - a, -b:ny - b, -c:nz - c]
    return x, y, z


def get_tight_ball_section(radius: float, center: tuple or list or np.array, shape: tuple) -> np.array:
    radius = int(radius)
    y, x, z = get_grid_around_origin(shape=shape, center=center)
    ball = 1 * (x * x + y * y + z * z <= radius * radius)
    return ball


def get_subgrid_around_point(shape: tuple, center: tuple or list or np.array, shift: tuple):
    """
    :param shift:
    :param shape:
    :param center:
    :return:
    """
    # nx, ny, nz = shape
    dxl, dyl, dzl = np.array(shift, dtype=int)
    dxr, dyr, dzr = np.array(shape, dtype=int) - shift
    # print("dxl, dyl, dzl = {}, dxr, dyr, dzr = {}".format((dxl, dyl, dzl), (dxr, dyr, dzr)))
    a, b, c = center
    y, x, z = np.ogrid[a - dxl:dxr + a, b - dyl:dyr + b, c - dzl:dzr + c]
    return x, y, z


def get_wij3_efficient(domain: np.array, xi: tuple or list or np.array, radius: float) -> float:
    radius = int(radius)
    a, b, c = xi
    shx, shy, shz = domain.shape
    dxl, dyl, dzl = np.min([radius, a]), np.min([radius, b]), np.min([radius, c])
    dxr, dyr, dzr = np.min([radius, shx - a]), np.min([radius, shy - b]), np.min([radius, shz - c])
    nx, ny, nz = dxl + dxr, dyl + dyr, dzl + dzr

    ball = get_tight_ball_section(radius=radius, center=(dxl, dyl, dzl), shape=(nx, ny, nz))
    ball_vol = (4 * np.pi * radius ** 3) / 3
    y, x, z = get_subgrid_around_point(shape=ball.shape, center=xi, shift=(dxl, dyl, dzl))
    square_subdomain = domain[x, y, z]
    intersection = np.sum(ball * square_subdomain)
    wij = intersection / ball_vol
    return wij


def get_ripley_k3est_efficient(domain, pp, r_minimum, r_maximum, steps, corrections: None or str = "ripley") -> Tuple:
    """
    The Ripley estimate (accounting for edge corrections) is:

    K(t) ~ vol(D) np.sum( (|B_t(xi) && PP| - 1) / w_ij ) / n /(n-1),

    where w_ij = vol(B_rij(xi) && D) and:

    && stands for domain intersection, D = domain, |.| is the cardinality,
    vol() is the volume.

    :return:
    :param domain:
    :param pp:
    :param r_minimum:
    :param r_maximum:
    :param steps:
    :param corrections:
    :return:
    """
    # domain = np.pad(array=domain, pad_width=r_maximum)
    domain_vol = np.sum(domain)
    assert domain_vol > 0, "The domain is empty."
    n = len(pp)
    d = cdist(pp, pp)
    d_sorted = np.sort(d, axis=1)[:, 1:]
    assert np.min(d_sorted) > 0, "Some points in the list are repeated, " \
                                 "introduce a filter"

    # Calculate edge correctors w_ij
    if corrections == "ripley":
        w = np.zeros(d_sorted.shape)
        for i, xi in enumerate(pp):
            # xi_shifted = np.array(xi, dtype=int) + r_maximum * np.ones(3, dtype=int)
            for j in range(n - 1):
                dij = d_sorted[i, j]
                if dij <= r_maximum:
                    if dij <= 3:
                        wij = 1
                    else:
                        wij = get_wij3_efficient(domain=domain, xi=xi, radius=dij)
                        assert wij > 0, "Point xi={} out from the domain! dij = {}".format(xi, dij)
                    w[i, j] = wij
    else:
        w = np.ones(d_sorted.shape)

    k_ripley = []
    r_range = np.linspace(start=r_minimum, stop=r_maximum, num=steps)
    for t in r_range:
        k_t = 0
        for i in range(n):
            ri = d_sorted[i, :]
            indicator = (ri <= t)
            i_t = np.sum(1 * indicator) - 1
            if i_t > 0:
                wij = np.min(w[i, indicator])
                k_t += i_t / wij
        k_ripley.append(k_t)

    k_ripley = domain_vol * np.array(k_ripley) / n / (n - 1)

    return k_ripley, r_range


Sp_WT = [
    # "180426/004",
    # "180426/005",
    "180426/021", "180426/024", "180711/003",
    # "180711/004",
    # "180711/005",
    # "180711/018",
]
Sp_ED1h = ["181119/002", "181119/030", "181126/002", "181126/012", "181126/025"]
Sc_ED6h = ["ScED_6h/001", "ScED_6h/002", "ScED_6h/003", "ScED_6h/006", "ScED_6h/011"]
Sc_ED1h = ["190329/001", "190329/004", "190329/005", "190329/007", "190329/010", "190329/012", "190329/013",
           "190329/015", "190329/017", "190329/021", "190329/022", "190329/023", "190329/025", "190329/028",
           "190329/032", "190329/036", ]
# FOR_RIBO_CONCENTRATION = [
#     "190301/001", "190301/002", "190301/004", "190301/006", "190301/007", "190301/010", "190301/011", "190301/013",
#     "190301/014", "190301/015", "190301/017", "190301/018", "190301/019", "190301/020", "190301/021", "190301/026",
#     "190301/029", "190301/030", ]
sample_type = "Sc_ED1h"
if sample_type == "Sc_ED1h":
    tomos = Sc_ED1h
elif sample_type == "Sc_ED6h":
    tomos = Sc_ED6h
dataset_table = "/struct/mahamid/Irene/yeast/yeast_table.csv"
# tomo_name = "180426/005"

particle_radius = {"ribo": 8, "fas": 12}
binning = 4
resolution_ori_A = 3.37
voxel_size_nm = binning * resolution_ori_A * 0.1
voxel_size_microm = binning * resolution_ori_A * 0.0001

test = False
calculate_K_function_Ripley_est = False
calculate_K_function2 = False
calculate_K_function = False
calculate_distance_to_mask = False
calculate_distance_between_particles = True
calculate_points_density = False
calculate_envelope = False

if calculate_distance_between_particles:
    particles = ["ribo"]
    # particles = ["fas"]
    # particles = ["ribo", "fas"]
    particles_str = ""
    for particle in particles:
        particles_str += "_" + str(particle)
    stats_file_name = sample_type + "_distance_statistics" + particles_str + ".csv"
    statistics_file = os.path.join("/struct/mahamid/Irene/yeast/quantifications", stats_file_name)

    for tomo_name in tomos:
        motl_file_name = os.path.join("/struct/mahamid/Irene/yeast/quantifications", tomo_name)
        os.makedirs(motl_file_name, exist_ok=True)
        motl_file_name = os.path.join(motl_file_name, "ribo_motl_with_nn_dist_score.csv")
        coordinates_lists = []
        for particle in particles:
            motl_path = extract_motl(particle=particle, dataset_table=dataset_table, tomo_name=tomo_name)
            print(particle, motl_path)
            _, motl_coords = read_motl_coordinates_and_values(path_to_motl=motl_path)
            coordinates_lists.append(motl_coords)
        if len(coordinates_lists) > 1:
            coordinates1, coordinates2 = coordinates_lists
            distance_distribution = get_minimum_distance_distribution_between_motls(coordinates1=coordinates1,
                                                                                    coordinates2=coordinates2,
                                                                                    voxel_size=voxel_size_nm)
        else:
            coordinates1 = coordinates_lists[0]
            distance_distribution = get_minimum_distance_distribution_between_motls(coordinates1=coordinates1,
                                                                                    voxel_size=voxel_size_nm)
        # from file_actions.writers.h5 import write_dataset_hdf
        # from networks.utils import build_prediction_output_dir
        # from tomogram_utils.coordinates_toolbox.clustering import get_cluster_centroids

        motive_list_df = build_tom_motive_list(
            list_of_peak_coordinates=coordinates1,
            list_of_peak_scores=distance_distribution, in_tom_format=True)
        motive_list_df.to_csv(motl_file_name, index=False, header=False)
        print("Motive list saved in", motl_file_name)
        write_distribution(statistics_file=statistics_file, tomo_name=tomo_name, stat_measure=distance_distribution)

    # stats_df = pd.read_csv(statistics_file)
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    #
    # sns.swarmplot(data=stats_df, palette='Blues', size=5)
    # sns.boxplot(data=stats_df, color='white')
    # plt.xlabel('tomo_name')
    # plt.show()

if calculate_distance_to_mask:
    mask_class = "memb"
    particle = "fas"

    particles_str = mask_class + "_" + particle

    stats_file_name = sample_type + "_distance_statistics_" + particles_str + ".csv"
    statistics_file = os.path.join("/struct/mahamid/Irene/yeast/quantifications", stats_file_name)

    if sample_type == "healthy":
        tomos = HEALTHY
    else:
        tomos = ED

    for tomo_name in tomos:
        distance_distribution = get_distance_motl_to_mask(dataset_table=dataset_table, tomo_name=tomo_name,
                                                          particle=particle, mask=mask_class, voxel_size=voxel_size_nm)
        write_distribution(statistics_file=statistics_file, tomo_name=tomo_name, stat_measure=distance_distribution)

if calculate_points_density:
    mask_class = "cytosol"
    particle = "fas"
    statistics_file = "/struct/mahamid/Irene/yeast/quantifications/" + particle + "_concentration_statistics.csv"
    for tomo_name in ED:
        stats_label = particle + "_per_micrm"
        particle_concentration = get_points_density_within_mask(dataset_table=dataset_table, tomo_name=tomo_name,
                                                                particle=particle, mask=mask_class,
                                                                voxel_size=voxel_size_microm)

        print("{} concentration is {} particles/(micrometer)^3.".format(particle, particle_concentration))
        write_quantification_value(statistics_file=statistics_file, statistics_label=stats_label,
                                   tomo_name=tomo_name, stat_measure=particle_concentration)
    # stats_df = pd.read_csv(statistics_file)

    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # 
    # sns.swarmplot(y=distance_distribution, palette='Blues', size=5)
    # # sns.boxplot(data=stats_df, color='white')
    # plt.xlabel('tomo_name')
    # plt.show()

if calculate_K_function:
    """
    For a stationary point process Phi in three-dimensional space, the three-dimensional K function is

    K3(r) = (1/lambda) E(N(Phi,x,r) | x in Phi)
    
    where 
    lambda is the intensity of the process (the expected number of points per unit volume) and 
    N(Phi,x,r) is the number of points of Phi, other than x itself, which fall within a distance r of x. 
    This is the three-dimensional generalisation of Ripley's K function for two-dimensional point processes 
    (Ripley, 1977).
    
    The three-dimensional point pattern X is assumed to be a partial realisation of a stationary point process Phi. 
    The distance between each pair of distinct points is computed. 
    The empirical cumulative distribution function of these values, with appropriate edge corrections, 
    is renormalised to give the estimate of K3(r).
    
    The available edge corrections are:
    
    "translation":
    the Ohser translation correction estimator (Ohser, 1983; Baddeley et al, 1993)
    
    "isotropic":
    the three-dimensional counterpart of Ripley's isotropic edge correction (Ripley, 1977; Baddeley et al, 1993).
    
    Alternatively correction="all" selects all options.
    """
    mask = "cytosol"
    particle = "ribo"
    mask_value = 1

    print("Calculating K function for tomo", tomo_name)

    df = pd.read_csv(dataset_table)
    DTHeader_masks = DatasetTableHeader(semantic_classes=[mask])
    DTHeader_particles = DatasetTableHeader(semantic_classes=[particle])
    tomo_row = df.loc[df[DTHeader_masks.tomo_name] == tomo_name]
    mask_column = DTHeader_masks.masks_names[0]
    mask_path = tomo_row[mask_column].values[0]
    print(mask_path)
    mask_array = load_tomogram(mask_path)
    mask_volume_nm = np.sum(mask_array) * (voxel_size_nm ** 3)
    assert mask_volume_nm != 0, "The domain mask is empty."

    motl_column = DTHeader_particles.clean_motls[0]
    motl_path = tomo_row[motl_column].values[0]
    _, motl_coords = read_motl_coordinates_and_values(path_to_motl=motl_path)
    # #TODO rm following line
    # motl_coords = list(motl_coords)
    # motl_coords = [[10,10,10]] + motl_coords

    coords = np.array([[p[2], p[1], p[0]] for p in motl_coords])

    n = len(coords)
    lambda_poisson = n / mask_volume_nm
    # pp3 = np.zeros(mask_array.shape, dtype=int)
    # pp3[tuple(coords.T)] = 1
    d = cdist(coords, coords)
    d_mod = d + np.max(d) * np.eye(d.shape[0])

    start = time()
    r = 40
    weighted_points = []
    for center, d_row in zip(coords[:10], list(d_mod)):
        # for center in coords:
        Vr = generate_ball_mask(shape=mask_array.shape, center=center, radius=r)
        # points_within_radius = np.sum(pp3 * Vr)
        Vr_ratio = calculate_ball_intersection_ratio(domain=mask_array, ball_mask=Vr)
        points_within_radius = len(np.where(d_row < r)[0])
        weighted_points.append(Vr_ratio * points_within_radius)

    K_r = lambda_poisson * np.sum(weighted_points) / n
    # K, list_of_radius_pix = K_estimate(points=coordinates_in_mask, segmentation_points=segmentation_points)
    end = time()
    print("elapsed time: {} secs.".format(end - start))
    # list_of_radius = [voxel_size_nm * radius for radius in list_of_radius_pix]
    # plt.plot(list_of_radius[:20], K[:20], "--*")
    # plt.savefig("/struct/mahamid/Irene/3d-cnn/K_estimate_80_nm.png")
if calculate_K_function2:
    """
    For a stationary point process Phi in three-dimensional space, the three-dimensional K function is

    K3(r) = (1/lambda) E(N(Phi,x,r) | x in Phi)
    
    where 
    lambda is the intensity of the process (the expected number of points per unit volume) and 
    N(Phi,x,r) is the number of points of Phi, other than x itself, which fall within a distance r of x. 
    This is the three-dimensional generalisation of Ripley's K function for two-dimensional point processes 
    (Ripley, 1977).
    
    The three-dimensional point pattern X is assumed to be a partial realisation of a stationary point process Phi. 
    The distance between each pair of distinct points is computed. 
    The empirical cumulative distribution function of these values, with appropriate edge corrections, 
    is renormalised to give the estimate of K3(r).
    
    The available edge corrections are:
    
    "translation":
    the Ohser translation correction estimator (Ohser, 1983; Baddeley et al, 1993)
    
    "isotropic":
    the three-dimensional counterpart of Ripley's isotropic edge correction (Ripley, 1977; Baddeley et al, 1993).
    
    Alternatively correction="all" selects all options.
    """
    mask = "cytosol"
    particle = "ribo"
    mask_value = 1

    print("Calculating K function for tomo", tomo_name)

    df = pd.read_csv(dataset_table)
    DTHeader_masks = DatasetTableHeader(semantic_classes=[mask])
    DTHeader_particles = DatasetTableHeader(semantic_classes=[particle])
    tomo_row = df.loc[df[DTHeader_masks.tomo_name] == tomo_name]
    mask_column = DTHeader_masks.masks_names[0]
    mask_path = tomo_row[mask_column].values[0]
    print(mask_path)
    mask_array = load_tomogram(mask_path)
    mask_volume_nm = np.sum(mask_array) * (voxel_size_nm ** 3)
    assert mask_volume_nm != 0, "The domain mask is empty."
    mask_array_coords = list(np.transpose(np.where(mask_array == 1)))
    r_max_nm = 35
    num = 30
    r_max = r_max_nm / voxel_size_nm

    motl_column = DTHeader_particles.clean_motls[0]
    motl_path = tomo_row[motl_column].values[0]
    _, motl_coords = read_motl_coordinates_and_values(path_to_motl=motl_path)

    coords = np.array([[p[2], p[1], p[0]] for p in motl_coords])
    global_min_dist = np.min(cdist(coords, coords) + np.max(coords) * np.eye(len(coords)))
    r_min_nm = global_min_dist * voxel_size_nm
    n = len(coords)
    # lambda_poisson = n / mask_volume_nm
    Eest, radius_range = generate_Eest(coords=coords, r_min=global_min_dist, r_max=r_max, num=num)

    Envelope = []
    N_simulations = 100
    for t in tqdm(range(N_simulations)):
        k_sampled_points = random.sample(mask_array_coords, k=2 * n)
        sampled_points = [k_sampled_points[0]]

        for point in k_sampled_points[1:]:
            if len(sampled_points) < n:
                candidate_coords = np.append(sampled_points, [point], axis=0)
                dist = cdist(candidate_coords, candidate_coords)
                distance = np.min(dist + np.max(dist) * np.eye(len(candidate_coords)))
                if distance >= global_min_dist:
                    sampled_points = np.append(sampled_points, [point], axis=0)
        Eest_env, _ = generate_Eest(coords=sampled_points, r_min=global_min_dist, r_max=r_max, num=num)
        Envelope.append(Eest_env)

    envelope = np.array(Envelope)  # (n_envelopes, n_radius)
    # envelope_max = np.max(envelope, axis=0)
    # envelope_min = np.min(envelope, axis=0)
    sorted_envelope = np.sort(envelope, axis=0)
    percent_1 = int(N_simulations * 0.01)
    envelope_max = sorted_envelope[-percent_1, :]
    envelope_min = sorted_envelope[percent_1, :]
    envelope_mean = 0.5 * (envelope_min + envelope_max)
    fig, axs = plt.subplots()
    axs.plot(voxel_size_nm * np.array(radius_range), 0 * envelope_mean, "-", color='lightgreen')
    axs.fill_between(voxel_size_nm * np.array(radius_range), envelope_min - envelope_mean, envelope_max - envelope_mean,
                     color='lightgreen', alpha=0.5)
    axs.plot(voxel_size_nm * np.array(radius_range), np.array(Eest) - envelope_mean, "--o", color='cornflowerblue')
    title = "K function estimation for " + particle
    axs.set_title(title)
    axs.set_xlabel("radius (nm)")
    axs.set_xlim([r_min_nm, r_max_nm])
    axs.set_ylabel("K")
    labels = ["$K_{est} - mean(K_{CSR})$",
              "$K_{CSR} - mean(K_{CSR})$ envelope at 99%"]
    handles = [Line2D(xdata=[0], ydata=[0], color='cornflowerblue', marker='o'),
               plt.Rectangle((0, 0), 0.5, 0.1, color='lightgreen', alpha=0.5)]

    axs.legend(handles, labels, loc="upper right", shadow=True)
    # axs.legend((l1, l3, l2), ('K_est', 'K random mean', 'K random envelope at 95%'), loc="lower right", shadow=True)
    plt.savefig("/struct/mahamid/Irene/3d-cnn/K_estimate_centered_" + particle + "_" + str(r_max_nm) + "_rmax_nm.png")

    # fig, axs = plt.subplots()
    # axs.fill_between(voxel_size_nm * np.array(radius_range), envelope_min, envelope_max, color='lightgreen', alpha=0.5)
    # axs.plot(voxel_size_nm * np.array(radius_range), Eest, "--o", color='cornflowerblue')
    # axs.plot(voxel_size_nm * np.array(radius_range), envelope_mean, "-", color='lightgreen')
    # title = "K function estimation for " + particle
    # axs.set_title(title)
    # axs.set_xlabel("radius (nm)")
    # axs.set_xlim([r_min_nm, r_max_nm])
    # axs.set_ylabel("K")
    # labels = ["K_est",
    #           "$K_{CSR}$ envelope at 95%",
    #           "$K_{CSR}$ mean"]
    # handles = [Line2D(xdata=[0], ydata=[0], color='cornflowerblue', marker='o'),
    #            plt.Rectangle((0, 0), 0.5, 0.1, color='lightgreen', alpha=0.5),
    #            Line2D(xdata=[0], ydata=[0], color='lightgreen')]
    #
    # axs.legend(handles, labels, loc="lower right", shadow=True)
    # # axs.legend((l1, l3, l2), ('K_est', 'K random mean', 'K random envelope at 95%'), loc="lower right", shadow=True)
    # plt.savefig("/struct/mahamid/Irene/3d-cnn/K_estimate_" + particle + ".png")
if calculate_K_function_Ripley_est:
    """
    For a stationary point process Phi in three-dimensional space, the three-dimensional K function is

    K3(r) = (1/lambda) E(N(Phi,x,r) | x in Phi)
    
    where 
    lambda is the intensity of the process (the expected number of points per unit volume) and 
    N(Phi,x,r) is the number of points of Phi, other than x itself, which fall within a distance r of x. 
    This is the three-dimensional generalisation of Ripley's K function for two-dimensional point processes 
    (Ripley, 1977).
    
    The three-dimensional point pattern X is assumed to be a partial realisation of a stationary point process Phi. 
    The distance between each pair of distinct points is computed. 
    The empirical cumulative distribution function of these values, with appropriate edge corrections, 
    is renormalised to give the estimate of K3(r).
    
    The available edge corrections are:
    
    "translation":
    the Ohser translation correction estimator (Ohser, 1983; Baddeley et al, 1993)
    
    "isotropic":
    the three-dimensional counterpart of Ripley's isotropic edge correction (Ripley, 1977; Baddeley et al, 1993).
    
    Alternatively correction="all" selects all options.
    """
    mask = "cytosol"
    particle = "ribo"
    sample_type = "healthy"
    mask_value = 1

    df = pd.read_csv(dataset_table)
    DTHeader_masks = DatasetTableHeader(semantic_classes=[mask])
    DTHeader_particles = DatasetTableHeader(semantic_classes=[particle])
    r_max_nm = 50
    r_max = int(r_max_nm / voxel_size_nm)
    r_min = 3  # 2 * particle_radius[particle]
    r_min_nm = r_min * voxel_size_nm
    num = 30
    N_simulations = 100
    n_total = 0
    K_global = np.zeros(num)
    global_envelope_max = 0
    global_envelope_min = 0
    percent = 95
    import matplotlib as mpl

    mpl.use('Agg')
    for tomo_name in HEALTHY:
        print("Calculating K function for tomo", tomo_name)
        tomo_row = df.loc[df[DTHeader_masks.tomo_name] == tomo_name]
        mask_column = DTHeader_masks.masks_names[0]
        mask_path = tomo_row[mask_column].values[0]
        print(mask_path)
        mask_array = load_tomogram(mask_path)
        mask_array_coords = list(np.transpose(np.where(mask_array == 1)))
        print("number of vals in mask", len(mask_array_coords))
        mask_array_coords = [tuple(point) for point in mask_array_coords]
        mask_volume_nm = np.sum(mask_array) * (voxel_size_nm ** 3)
        assert mask_volume_nm != 0, "The domain mask is empty."

        motl_column = DTHeader_particles.clean_motls[0]
        motl_path = tomo_row[motl_column].values[0]
        _, motl_coords = read_motl_coordinates_and_values(path_to_motl=motl_path)
        coords = [(p[2], p[1], p[0]) for p in motl_coords]
        coords = set(coords) & set(mask_array_coords)
        coords = np.array(list(coords))
        n = len(coords)
        print("number odf coords in mask", n)

        coords = np.array(coords)
        global_min_dist = np.min(cdist(coords, coords) + np.max(coords) * np.eye(len(coords)))

        particle_diam_nm = r_min * voxel_size_nm

        """
        Here to calculate the edge correctors
        """
        start = time()
        Kest, radius_range = get_ripley_k3est_efficient(domain=mask_array, pp=coords, r_minimum=r_min,
                                                        r_maximum=r_max, steps=num)
        end = time()
        print("Elapsed time {} secs.".format(end - start))

        Envelope = []

        for t in tqdm(range(N_simulations)):
            sampled_points = random.sample(mask_array_coords, k=n)
            Kest_env, _ = get_ripley_k3est_efficient(domain=mask_array, pp=sampled_points, r_minimum=r_min,
                                                     r_maximum=r_max, steps=num)
            # get_Ripley_Kest3(A=mask_array, coords=sampled_points, r_min=r_min, r_max=r_max,
            #                    num=num)
            distance_distribution = get_minimum_distance_distribution_between_motls(coordinates1=sampled_points,
                                                                                    voxel_size=voxel_size_nm)
            Envelope.append(Kest_env)

        envelope = np.array(Envelope)  # (n_envelopes, n_radius)
        sorted_envelope = np.sort(envelope, axis=0)

        percent_1 = int(N_simulations * 0.01 * (100 - percent))
        envelope_max = sorted_envelope[-percent_1, :]
        envelope_min = sorted_envelope[percent_1, :]
        envelope_mean = 0.5 * (envelope_min + envelope_max)

        fig, axs = plt.subplots(1, 2, sharex=True, figsize=(10, 5))
        axs[0].plot(voxel_size_nm * np.array(radius_range), envelope_mean, "-", color='lightgreen')
        axs[0].fill_between(voxel_size_nm * np.array(radius_range), envelope_min, envelope_max,
                            color='lightgreen', alpha=0.5)
        axs[0].plot(voxel_size_nm * np.array(radius_range), np.array(Kest), "--o", color='cornflowerblue')
        title = "Ripley's K estimate for " + particle + " (" + tomo_name + ")"
        axs[0].set_title(title)
        axs[0].set_xlabel("radius (nm)")
        axs[0].set_xlim([r_min_nm, r_max_nm])
        axs[0].set_ylabel("K")
        envelope_label = "$K_{CSR}$ envelope at " + str(percent) + "%"
        labels = ["$K_{obs}$",
                  envelope_label]
        handles = [Line2D(xdata=[0], ydata=[0], color='cornflowerblue', marker='*'),
                   plt.Rectangle((0, 0), 0.5, 0.1, color='lightgreen', alpha=0.5)]

        axs[0].legend(handles, labels, loc="upper right", shadow=True)

        axs[1].plot(voxel_size_nm * np.array(radius_range), 0 * envelope_mean, "-", color='lightgreen')
        axs[1].fill_between(voxel_size_nm * np.array(radius_range), envelope_min - envelope_mean,
                            envelope_max - envelope_mean,
                            color='lightgreen', alpha=0.5)
        axs[1].plot(voxel_size_nm * np.array(radius_range), np.array(Kest) - envelope_mean, "--o",
                    color='cornflowerblue')
        title = "Ripley's K estimate for " + particle + " (" + tomo_name + ")"
        axs[1].set_title(title)
        axs[1].set_xlabel("radius (nm)")
        axs[1].set_xlim([r_min_nm, r_max_nm])
        axs[1].set_ylabel("K")
        envelope_label = "$K_{CSR} - mean(K_{CSR})$ envelope at " + str(percent) + "%"
        labels = ["$K_{est} - mean(K_{CSR})$",
                  envelope_label]
        handles = [Line2D(xdata=[0], ydata=[0], color='cornflowerblue', marker='o'),
                   plt.Rectangle((0, 0), 0.5, 0.1, color='lightgreen', alpha=0.5)]

        axs[1].legend(handles, labels, loc="upper right", shadow=True)
        fig_file = os.path.join("/struct/mahamid/Irene/3d-cnn/FIGS", tomo_name)
        os.makedirs(fig_file, exist_ok=True)
        fig_file = os.path.join(fig_file,
                                "K_Ripley_estimate_" + str(percent) + "percent_" + particle + "_" + str(
                                    r_max_nm) + "_rmax_nm.png")
        plt.savefig(fig_file)
        print("saved plot in {}".format(fig_file))
        plt.close()

        # K_global += n * Kest
        # n_total += n
        # global_envelope_max += n * envelope_max
        # global_envelope_min += n * envelope_min
        # global_envelope.

    # global_envelope_max = global_envelope_max / n_total
    # global_envelope_min = global_envelope_min / n_total
    # global_envelope_mean = 0.5 * (global_envelope_min + global_envelope_max)
    #
    # fig, axs = plt.subplots()
    # axs.plot(voxel_size_nm * np.array(radius_range), global_envelope_mean, "-", color='lightgreen')
    # axs.fill_between(voxel_size_nm * np.array(radius_range), global_envelope_min,
    #                  global_envelope_max,
    #                  color='lightgreen', alpha=0.5)
    # axs.plot(voxel_size_nm * np.array(radius_range), np.array(K_global), "--o",
    #          color='cornflowerblue')
    # title = "Global K (Ripley's estimate): " + particle
    # axs.set_title(title)
    # axs.set_xlabel("radius (nm)")
    # axs.set_xlim([r_min_nm, r_max_nm])
    # axs.set_ylabel("K")
    # envelope_label = "$K_{CSR} - mean(K_{CSR})$ envelope at " + str(percent) + "%"
    # labels = ["$K_{est} - mean(K_{CSR})$",
    #           envelope_label]
    # handles = [Line2D(xdata=[0], ydata=[0], color='cornflowerblue', marker='o'),
    #            plt.Rectangle((0, 0), 0.5, 0.1, color='lightgreen', alpha=0.5)]
    #
    # axs.legend(handles, labels, loc="upper right", shadow=True)
    # fig_file = "/struct/mahamid/Irene/3d-cnn/FIGS"
    # os.makedirs(fig_file, exist_ok=True)
    # fig_file = os.path.join(fig_file, sample_type + "_K_Ripley_estimate_" + str(percent) +
    #                         "percent_" + particle + "_" + str(r_max_nm) + "_rmax_nm.png")
    # plt.savefig(fig_file)
    # plt.close()
    #
    # # Starting centered plot
    # fig, axs = plt.subplots()
    # axs.plot(voxel_size_nm * np.array(radius_range), 0 * global_envelope_mean, "-", color='lightgreen')
    # axs.fill_between(voxel_size_nm * np.array(radius_range), global_envelope_min - global_envelope_mean,
    #                  global_envelope_max - global_envelope_mean,
    #                  color='lightgreen', alpha=0.5)
    # axs.plot(voxel_size_nm * np.array(radius_range), np.array(K_global) - global_envelope_mean, "--o",
    #          color='cornflowerblue')
    # title = "Global K (Ripley's estimate): " + particle
    # axs.set_title(title)
    # axs.set_xlabel("radius (nm)")
    # axs.set_xlim([r_min_nm, r_max_nm])
    # axs.set_ylabel("K")
    # envelope_label = "$K_{CSR} - mean(K_{CSR})$ envelope at " + str(percent) + "%"
    # labels = ["$K_{est} - mean(K_{CSR})$",
    #           envelope_label]
    # handles = [Line2D(xdata=[0], ydata=[0], color='cornflowerblue', marker='o'),
    #            plt.Rectangle((0, 0), 0.5, 0.1, color='lightgreen', alpha=0.5)]
    #
    # axs.legend(handles, labels, loc="upper right", shadow=True)
    # fig_file = "/struct/mahamid/Irene/3d-cnn/FIGS"
    # os.makedirs(fig_file, exist_ok=True)
    # fig_file = os.path.join(fig_file, sample_type + "_Global_K_centered_Ripley_estimate_" + str(percent) +
    #                         "percent_" + particle + "_" + str(r_max_nm) + "_rmax_nm.png")
    # plt.savefig(fig_file)
# if calculate_envelope:
#     mask = "cytosol"
#     particle = "ribo"
#     sample_type = "healthy"
#     mask_value = 1
#
#     r_max_nm = 30
#     r_max = int(r_max_nm / voxel_size_nm)
#     r_min = 2 * particle_radius[particle]
#     r_min_nm = r_min * voxel_size_nm
#     num = 30
#     N_simulations = 10
#     n_total = 0
#     K_global = np.zeros(num)
#     global_envelope_max = 0
#     global_envelope_min = 0
#     percent = 95
#
#     if sample_type == "healthy":
#         tomos = HEALTHY
#     elif sample_type == "ED":
#         tomos = ED
#
#     df = pd.read_csv(dataset_table)
#     DTHeader_masks = DatasetTableHeader(semantic_classes=[mask])
#     DTHeader_particles = DatasetTableHeader(semantic_classes=[particle])
#
#     statistics_dir = os.path.join("/struct/mahamid/Irene/yeast/quantifications/NN", particle)
#     os.makedirs(statistics_dir, exist_ok=True)
#     for tomo_name in tomos:
#         print("Calculating envelope for NN: tomo", tomo_name)
#         tomo_row = df.loc[df[DTHeader_masks.tomo_name] == tomo_name]
#         mask_column = DTHeader_masks.masks_names[0]
#         mask_path = tomo_row[mask_column].values[0]
#         print(mask_path)
#         mask_array = load_tomogram(mask_path)
#         mask_array_coords = list(np.transpose(np.where(mask_array == mask_value)))
#
#         print("number of vals in mask", len(mask_array_coords))
#         mask_array_coords = [tuple(point) for point in mask_array_coords]
#         mask_volume_nm = np.sum(mask_array) * (voxel_size_nm ** 3)
#         assert mask_volume_nm != 0, "The domain mask is empty."
#
#         motl_column = DTHeader_particles.clean_motls[0]
#         motl_path = tomo_row[motl_column].values[0]
#         _, motl_coords = read_motl_coordinates_and_values(path_to_motl=motl_path)
#         coords = [(p[2], p[1], p[0]) for p in motl_coords]
#         coords = set(coords) & set(mask_array_coords)
#         coords = np.array(list(coords))
#         n = len(coords)
#         print("number odf coords in mask", n)
#
#         global_min_dist = np.min(cdist(coords, coords) + np.max(coords) * np.eye(len(coords)))
#         particle_diam_nm = r_min * voxel_size_nm
#
#         # start = time()
#         stats_file_name = "envelope_" + sample_type + "_distance_statistics.csv"
#         statistics_file = os.path.join(statistics_dir, stats_file_name)
#         # distance_distribution = \
#         #     get_minimum_distance_distribution_between_motls(coordinates1=coords, voxel_size=voxel_size_nm)
#         # write_distribution(statistics_file=statistics_file, tomo_name=tomo_name, stat_measure=distance_distribution)
#         # end = time()
#         # print("Elapsed time {} secs.".format(end - start))
#
#         for t in tqdm(range(N_simulations)):
#             sampled_points = random.sample(mask_array_coords, k=n)
#             tomo_name_env = tomo_name + str(t)
#             distance_distribution_env = get_minimum_distance_distribution_between_motls(coordinates1=sampled_points,
#                                                                                         voxel_size=voxel_size_nm)
#             write_distribution(statistics_file=statistics_file, tomo_name=tomo_name_env,
#                                stat_measure=distance_distribution_env)
