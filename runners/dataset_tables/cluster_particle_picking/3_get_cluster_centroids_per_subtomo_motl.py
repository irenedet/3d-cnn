import pandas as pd
import numpy as np
import argparse
from os import makedirs
from os.path import join

from src.python.filewriters.csv import build_tom_motive_list
from src.python.coordinates_toolbox.utils import average_duplicated_centroids, \
    get_cluster_centroids_from_partition

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

full_centroids_list = \
    get_cluster_centroids_from_partition(partition=partition,
                                         label_name=label_name,
                                         min_cluster_size=min_cluster_size,
                                         max_cluster_size=max_cluster_size,
                                         output_shape=output_shape,
                                         overlap=overlap,
                                         segmentation_class=class_number)

# Double-check centroids to avoid duplicates
unique_centroids = average_duplicated_centroids(
    motl_coords=full_centroids_list, min_peak_distance=particle_radius)

print("min point[2]", np.min([point[2] for point in unique_centroids]))
print("min point[1]", np.min([point[1] for point in unique_centroids]))
print("min point[0]", np.min([point[0] for point in unique_centroids]))

print("max point[2]", np.max([point[2] for point in unique_centroids]))
print("max point[1]", np.max([point[1] for point in unique_centroids]))
print("max point[0]", np.max([point[0] for point in unique_centroids]))

motl_name = "motl_" + str(len(unique_centroids)) + ".csv"
motl_file_name = join(output_dir, motl_name)

motive_list_df = build_tom_motive_list(
    list_of_peak_coordinates=unique_centroids, in_tom_format=False)
motive_list_df.to_csv(motl_file_name, index=False, header=False)
print("Motive list saved in", motl_file_name)
