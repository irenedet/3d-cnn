import argparse
from os import makedirs
from os.path import join, isfile

import numpy as np
import pandas as pd
import torch.nn as nn

from coordinates_toolbox.clustering import get_cluster_centroids
from filereaders.datasets import load_dataset
from filewriters.csv import build_tom_motive_list
from filewriters.h5 import write_dataset_hdf, \
    write_dataset_from_subtomos_with_overlap_multiclass
from naming import h5_internal_paths

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

clusters_output_path = join(output_dir, "clusters.hdf")

if not isfile(output_path):
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
        overlap, final_activation=nn.Sigmoid())

dataset = load_dataset(path_to_dataset=output_path)

clustering_labels, centroids_list, cluster_size_list = \
    get_cluster_centroids(dataset=dataset,
                          min_cluster_size=min_cluster_size,
                          max_cluster_size=max_cluster_size,
                          connectivity=1)

if not isfile(clusters_output_path):
    write_dataset_hdf(output_path=clusters_output_path,
                      tomo_data=clustering_labels)
# # Double-check centroids to avoid duplicates

motl_name = "motl_" + str(len(centroids_list)) + ".csv"
motl_file_name = join(output_dir, motl_name)

motive_list_df = build_tom_motive_list(
    list_of_peak_coordinates=centroids_list,
    list_of_peak_scores=cluster_size_list, in_tom_format=False)
motive_list_df.to_csv(motl_file_name, index=False, header=False)
print("Motive list saved in", motl_file_name)
