from os import makedirs
from os.path import join

import numpy as np
from scipy.spatial import distance

from coordinates_toolbox.utils import get_clusters_within_size_range
from filereaders.datasets import load_dataset
from filewriters.csv import build_tom_motive_list
from peak_toolbox.utils import read_motl_coordinates_and_values

global_output_dir = "/scratch/trueba/3d-cnn/cnn_evaluation/liang_dataset/"
# global_output_dir = "/home/papalotl/Desktop/"
particle_radius = 25
volume_sphere = 4 / 3 * np.pi * (particle_radius ** 3)

print("The volume of a sphere with radius", particle_radius, "is",
      volume_sphere)

min_cluster_size = volume_sphere // 20  # to be checked
max_cluster_size = 15 * volume_sphere  # to be checked
connectivity = 1  # to be checked
threshold = 1.5  # to be checked

output_shape = (450, 928, 928)
overlap = 12

peaks_motl_path = "/scratch/trueba/3d-cnn/cnn_evaluation/liang_dataset/NO_DA_ribo_D_2_IF_8/246/class_0/in_lamella/motl_818.csv"
output_dir = "/scratch/trueba/3d-cnn/cnn_evaluation/liang_dataset/clusters/test/clustering_test_NO_DA_ribo_D_2_IF_8_pr_radius_20/full_dataset/246/class_0"
output_path = join(output_dir, "prediction.hdf")

subfolder_name = "combined_motl_" + str(threshold) + "sph"
output_dir = join(output_dir, subfolder_name)
makedirs(name=output_dir, exist_ok=True)

# clusters_output_path = join(output_dir, "clusters_conn_1.hdf")
big_clusters_motl_path = join(output_dir, "big_clusters_motl.csv")
centroids_motl_path = join(output_dir, "small_clusters_motl.csv")
combined_motl_path = join(output_dir, "combined_motl.csv")

# dataset = load_dataset(path_to_dataset=output_path)
# print(dataset.shape)
dataset = load_dataset(path_to_dataset=output_path)
print("loaded dataset")
print(dataset.shape)

labeled_clusters, labels_list_within_range, cluster_size_within_range = \
    get_clusters_within_size_range(dataset=dataset,
                                   min_cluster_size=min_cluster_size,
                                   max_cluster_size=max_cluster_size,
                                   connectivity=connectivity)
print("cluster_size_within_range =", cluster_size_within_range)

_, predicted_coordinates = read_motl_coordinates_and_values(
    path_to_motl=peaks_motl_path)

# split into big and small clusters: ToDo define splitting parameter

small_clusters_labels = labels_list_within_range[
    np.array(cluster_size_within_range) < volume_sphere * threshold]
big_clusters_labels = labels_list_within_range[
    np.array(cluster_size_within_range) > volume_sphere * threshold]

# # get centroids of small clusters
centroids_list = list()
for index, label in enumerate(small_clusters_labels):
    print("label index = ", index)
    cluster = np.where(labeled_clusters == label)
    centroid = np.rint(np.mean(cluster, axis=1))
    centroids_list.append(centroid)
print("Number of small clusters in tomogram =", len(centroids_list))

# return labeled_clusters, centroids_list, cluster_size_within_range

coordinates = list()
# # for big clusters:
print("Number of big clusters:", len(big_clusters_labels))
peaks = np.array([[p[2], p[1], p[0]] for p in predicted_coordinates])

for index, label in enumerate(big_clusters_labels):
    print("label index = ", index)
    cluster = np.array(np.where(labeled_clusters == label))
    cluster = cluster.transpose()
    # print("cluster.shape", cluster.shape)
    distance_to_cluster = distance.cdist(peaks, cluster).min(axis=1)
    # print("distance_to_cluster", distance_to_cluster[0])
    if np.min(distance_to_cluster) == 0:
        peaks_on_cluster = list(peaks[distance_to_cluster == 0])
        print("peaks_on_cluster", len(peaks_on_cluster))
        coordinates += peaks_on_cluster

# motl_centroids = pd.read_csv(motive_list_centroids, header=None)
# motl_peaks = pd.read_csv(motive_list_peaks, header=None)
peaks_in_big_clusters_motl = build_tom_motive_list(
    list_of_peak_coordinates=coordinates,
    in_tom_format=False)
peaks_in_big_clusters_motl.to_csv(big_clusters_motl_path, index=False,
                                  header=False)
print("Motive list saved in", big_clusters_motl_path)

if len(centroids_list) > 0:
    small_clusters_centroids_motl = build_tom_motive_list(
        list_of_peak_coordinates=centroids_list,
        in_tom_format=False)
    small_clusters_centroids_motl.to_csv(centroids_motl_path, index=False,
                                         header=False)
    print("Motive list saved in", centroids_motl_path)

    combined_motl = peaks_in_big_clusters_motl.append(
        small_clusters_centroids_motl,
        sort=False)
    combined_motl.to_csv(combined_motl_path, header=False, index=False)
else:
    print("Only large clusters found...")
    peaks_in_big_clusters_motl.to_csv(combined_motl_path, index=False,
                                      header=False)
