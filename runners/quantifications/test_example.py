"""
This test is to compare to the R package spacestat? check name

x,y,z
0.6521905,0.3458780,0.6362098
0.5634816,0.5329997,0.5668492
0.6401469,0.6577303,0.6083227
0.4531872,0.4986831,0.5561061
0.4024065,0.5185939,0.4619537
0.4301314,0.5978692,0.4211341
0.5426670,0.3331305,0.6086559
0.5171540,0.4812385,0.4693289
0.5380821,0.5346181,0.6895762
0.6247595,0.5202695,0.4438619
0.5994358,0.4472798,0.4572158
0.6273128,0.4021228,0.3137526
0.5677997,0.5578387,0.4877353
0.5252906,0.4208006,0.5611905
0.5136530,0.5119949,0.5130261
0.3823755,0.3934489,0.8786302
0.4986575,0.7219554,0.5572538
0.6813928,0.4622317,0.4806036
0.5067576,0.6041771,0.3421154
0.6914811,0.5709185,0.4779528
0.5608215,0.4946501,0.4546417
0.5527819,0.6112838,0.5054802
0.3277538,0.5544602,0.4965354
0.5056904,0.4993069,0.3749173
0.4495825,0.6301057,0.4502732
0.4191831,0.4907028,0.5276468
0.4328677,0.5455024,0.4692569
0.5286325,0.4940597,0.4381875
0.5176031,0.5500188,0.5851076
0.5471704,0.4619051,0.2904810
0.5084539,0.3238646,0.5867900
0.3650219,0.6702248,0.5161296
0.4470632,0.5659267,0.5321392
0.5500109,0.5316267,0.5529418
0.4174284,0.4196340,0.3454396
0.3122635,0.5222964,0.6946390
0.4182125,0.5570697,0.4650019
0.4443124,0.4404332,0.5731256
0.5023129,0.4109328,0.5673851
"""
import os
import random
from time import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.signal import convolve
from scipy.spatial.distance import cdist
from tqdm import tqdm

from constants.dataset_tables import DatasetTableHeader
from file_actions.readers.tomograms import load_tomogram
from tomogram_utils.peak_toolbox.utils import _generate_unit_particle
from tomogram_utils.peak_toolbox.utils import read_motl_coordinates_and_values

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


def get_Ripley_Kest3(A, coords, r_min, r_max, num) -> Tuple:
    n = len(coords)
    d = cdist(coords, coords)
    r = np.sort(d, axis=1)[:, 1:]

    # print("Staring to generate edge correctors w!")
    w = np.ones(r.shape)
    for i, xi in enumerate(coords):
        for j in range(n - 1):
            rij = r[i, j]
            if rij <= r_max:
                wij = get_wij3(A=A, xi=xi, radius=rij)
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


dataset_table = ""
voxel_size = ""
mask = ""
particle = "ribo"
sample_type = "healthy"
mask_value = 1
voxel_size_nm = 1
particle_radius = 0
HEALTHY = ["test_001"]
df = pd.read_csv(dataset_table)
DTHeader_masks = DatasetTableHeader(semantic_classes=[mask])
DTHeader_particles = DatasetTableHeader(semantic_classes=[particle])
r_max_nm = 30
r_max = int(r_max_nm / voxel_size_nm)
r_min = 2 * particle_radius[particle]
r_min_nm = r_min * voxel_size_nm
num = 30
N_simulations = 100
n_total = 0
K_global = np.zeros(num)
global_envelope_max = 0
global_envelope_min = 0
percent = 95
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
    Kest, radius_range = get_Ripley_Kest3(A=mask_array, coords=coords, r_min=r_min, r_max=r_max, num=num)
    end = time()
    print("Elapsed time {} secs.".format(end - start))

    Envelope = []

    for t in tqdm(range(N_simulations)):
        k_sampled_points = random.sample(mask_array_coords, k=2 * n)
        sampled_points = [k_sampled_points[0]]

        for point in k_sampled_points[1:]:
            if len(sampled_points) < n:
                candidate_coords = np.append(sampled_points, [point], axis=0)
                dist = cdist(candidate_coords, candidate_coords)
                distance = np.min(dist + np.max(dist) * np.eye(len(candidate_coords)))
                if distance >= r_min:
                    sampled_points = np.append(sampled_points, [point], axis=0)
        Kest_env, _ = get_Ripley_Kest3(A=mask_array, coords=sampled_points, r_min=r_min, r_max=r_max,
                                       num=num)
        distance_distribution = get_minimum_distance_distribution_between_motls(coordinates1=sampled_points,
                                                                                voxel_size=voxel_size_nm)
        Envelope.append(Kest_env)

    envelope = np.array(Envelope)  # (n_envelopes, n_radius)
    sorted_envelope = np.sort(envelope, axis=0)

    percent_1 = int(N_simulations * 0.01 * (100 - percent))
    envelope_max = sorted_envelope[-percent_1, :]
    envelope_min = sorted_envelope[percent_1, :]
    envelope_mean = 0.5 * (envelope_min + envelope_max)
    fig, axs = plt.subplots()
    axs.plot(voxel_size_nm * np.array(radius_range), 0 * envelope_mean, "-", color='lightgreen')
    axs.fill_between(voxel_size_nm * np.array(radius_range), envelope_min - envelope_mean,
                     envelope_max - envelope_mean,
                     color='lightgreen', alpha=0.5)
    axs.plot(voxel_size_nm * np.array(radius_range), np.array(Kest) - envelope_mean, "--o", color='cornflowerblue')
    title = "Ripley's K estimate for " + particle + " (" + tomo_name + ")"
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
    fig_file = os.path.join("/struct/mahamid/Irene/3d-cnn/FIGS", tomo_name)
    os.makedirs(fig_file, exist_ok=True)
    fig_file = os.path.join(fig_file,
                            "K_centered_Ripley_estimate_" + str(percent) + "percent_" + particle + "_" + str(
                                r_max_nm) + "_rmax_nm.png")
    plt.savefig(fig_file)

    K_global += n * Kest
    n_total += n
    global_envelope_max += n * envelope_max
    global_envelope_min += n * envelope_min
    # global_envelope.
K_global = K_global / n_total
global_envelope_max = global_envelope_max / n_total
global_envelope_min = global_envelope_min / n_total
global_envelope_mean = 0.5 * (global_envelope_min + global_envelope_max)

fig, axs = plt.subplots()
axs.plot(voxel_size_nm * np.array(radius_range), 0 * global_envelope_mean, "-", color='lightgreen')
axs.fill_between(voxel_size_nm * np.array(radius_range), global_envelope_min - global_envelope_mean,
                 global_envelope_max - global_envelope_mean,
                 color='lightgreen', alpha=0.5)
axs.plot(voxel_size_nm * np.array(radius_range), np.array(K_global) - global_envelope_mean, "--o",
         color='cornflowerblue')
title = "Global K (Ripley's estimate): " + particle
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
fig_file = "/struct/mahamid/Irene/3d-cnn/FIGS"
os.makedirs(fig_file, exist_ok=True)
fig_file = os.path.join(fig_file, sample_type + "_Global_K_centered_Ripley_estimate_" + str(percent) +
                        "percent_" + particle + "_" + str(r_max_nm) + "_rmax_nm.png")
plt.savefig(fig_file)
