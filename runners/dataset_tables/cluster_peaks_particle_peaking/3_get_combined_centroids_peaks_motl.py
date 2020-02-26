import argparse
from os import makedirs
from os.path import join

import numpy as np
import pandas as pd
import torch.nn as nn
from scipy.spatial import distance

from coordinates_toolbox.clustering import get_clusters_within_size_range
from filereaders.datasets import load_dataset
from filewriters.csv import build_tom_motive_list
from filewriters.h5 import write_dataset_from_subtomos_with_overlap_multiclass
from naming import h5_internal_paths
from peak_toolbox.utils import read_motl_coordinates_and_values

parser = argparse.ArgumentParser()

parser.add_argument("-output", "--output_dir",
                    help="directory where the outputs will be stored",
                    type=str)
parser.add_argument("-label", "--label_name",
                    help="name of category to be segmented",
                    type=str)
parser.add_argument("-dataset_table", "--dataset_table",
                    help="path to dataset_table",
                    type=str)
parser.add_argument("-tomo_name", "--tomo_name",
                    help="tomo to be analyzed",
                    type=str)
parser.add_argument("-box", "--box_side",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-overlap", "--overlap",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-class_number", "--class_number",
                    help="class number associated to motifs list",
                    type=int)
parser.add_argument("-particle_radius", "--particle_radius",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-min_cluster_size", "--min_cluster_size",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-max_cluster_size", "--max_cluster_size",
                    help="name of category to be segmented",
                    type=int)
# New
parser.add_argument("-peaks_motl_path", "--peaks_motl_path",
                    help="peaks_motl_path",
                    type=str)
parser.add_argument("-cluster_size_threshold", "--cluster_size_threshold",
                    help="cluster_size_threshold",
                    type=int)

args = parser.parse_args()
output_dir = args.output_dir
label_name = args.label_name
dataset_table = args.dataset_table
tomo_name = args.tomo_name
box_side = args.box_side
overlap = args.overlap
class_number = args.class_number
particle_radius = args.particle_radius
min_cluster_size = args.min_cluster_size
max_cluster_size = args.max_cluster_size
# NEW
peaks_motl_path = args.peaks_motl_path
cluster_size_threshold = args.cluster_size_threshold

df = pd.read_csv(dataset_table)
df['tomo_name'] = df['tomo_name'].astype(str)
tomo_df = df[df['tomo_name'] == tomo_name]
z_shift = int(tomo_df.iloc[0]['z_shift'])
x_dim = int(tomo_df.iloc[0]['x_dim'])
y_dim = int(tomo_df.iloc[0]['y_dim'])
z_dim = int(tomo_df.iloc[0]['z_dim'])
partition = tomo_df.iloc[0]['test_partition']
print(partition)

subtomo_shape = tuple(box_side * np.array([1, 1, 1]))
output_shape = (z_dim, y_dim, x_dim)
makedirs(name=output_dir, exist_ok=True)

output_path = join(output_dir, "prediction.hdf")

subtomos_internal_path = join(
    h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS,
    label_name)

write_dataset_from_subtomos_with_overlap_multiclass(
    output_path,
    partition,
    output_shape,
    subtomo_shape,
    subtomos_internal_path,
    class_number,
    overlap,
    nn.Sigmoid())

dataset = load_dataset(path_to_dataset=output_path)
print("loaded dataset")
print(dataset.shape)
############################

connectivity = 1  # to be checked
# volume_sphere = 4 / 3 * np.pi * (particle_radius ** 3)
# print("The volume of a sphere with radius", particle_radius, "is",
#       volume_sphere)


subfolder_name = "combined_motl_" + str(cluster_size_threshold)
output_dir = join(output_dir, subfolder_name)
makedirs(name=output_dir, exist_ok=True)

big_clusters_motl_path = join(output_dir, "big_clusters_motl.csv")
centroids_motl_path = join(output_dir, "small_clusters_motl.csv")
combined_motl_path = join(output_dir, "combined_motl.csv")

labeled_clusters, labels_list_within_range, cluster_size_within_range = \
    get_clusters_within_size_range(dataset=dataset,
                                   min_cluster_size=min_cluster_size,
                                   max_cluster_size=max_cluster_size,
                                   connectivity=connectivity)
print("cluster_size_within_range =", cluster_size_within_range)

_, predicted_coordinates = read_motl_coordinates_and_values(
    path_to_motl=peaks_motl_path)

# split into big and small clusters:

small_clusters_labels = labels_list_within_range[
    np.array(cluster_size_within_range) < cluster_size_threshold]
big_clusters_labels = labels_list_within_range[
    np.array(cluster_size_within_range) > cluster_size_threshold]

# get centroids of small clusters
centroids_list = list()
for index, label in enumerate(small_clusters_labels):
    print("label index = ", index)
    cluster = np.where(labeled_clusters == label)
    centroid = np.rint(np.mean(cluster, axis=1))
    centroids_list.append(centroid)
print("Number of small clusters in tomogram =", len(centroids_list))

# for big clusters find intersecting peaks:
coordinates = list()
print("Number of big clusters:", len(big_clusters_labels))
peaks = np.array([[p[2], p[1], p[0]] for p in predicted_coordinates])

for index, label in enumerate(big_clusters_labels):
    print("label index = ", index)
    cluster = np.array(np.where(labeled_clusters == label))
    cluster = cluster.transpose()
    distance_to_cluster = distance.cdist(peaks, cluster).min(axis=1)
    if np.min(distance_to_cluster) == 0:
        peaks_on_cluster = list(peaks[distance_to_cluster == 0])
        print("peaks_on_cluster", len(peaks_on_cluster))
        coordinates += peaks_on_cluster

if len(coordinates) > 0:
    peaks_in_big_clusters_motl = build_tom_motive_list(
        list_of_peak_coordinates=coordinates,
        in_tom_format=False)
    peaks_in_big_clusters_motl.to_csv(big_clusters_motl_path, index=False,
                                      header=False)
    print("Motive list of big clusters saved in", big_clusters_motl_path)
else:
    print("No big clusters peaks!")

if len(centroids_list) > 0:
    small_clusters_centroids_motl = build_tom_motive_list(
        list_of_peak_coordinates=centroids_list,
        in_tom_format=False)
    small_clusters_centroids_motl.to_csv(centroids_motl_path, index=False,
                                         header=False)
    print("Motive list saved in", centroids_motl_path)
else:
    print("No centroids, i.e. no small clusters.")

if len(coordinates) > 0:
    if len(centroids_list) > 0:
        # We give preference to peaks, which are stored before:
        combined_motl = peaks_in_big_clusters_motl.append(
            small_clusters_centroids_motl,
            sort=False)
        combined_motl.to_csv(combined_motl_path, header=False, index=False)
    else:
        print("Only large clusters found...")
        peaks_in_big_clusters_motl.to_csv(combined_motl_path, index=False,
                                          header=False)

else:
    if len(centroids_list) > 0:
        print("Only small clusters found...")
        small_clusters_centroids_motl.to_csv(combined_motl_path, index=False,
                                             header=False)
    else:
        print("No peaks found!")
