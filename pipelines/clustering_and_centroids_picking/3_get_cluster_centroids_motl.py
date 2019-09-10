import sys
sys.path.insert(0, '/g/scb2/zaugg/trueba/3d-cnn/src')
print(sys.path)

from os import makedirs

from os.path import join
import numpy as np

from src.python.filewriters.csv import build_tom_motive_list
from src.python.coordinates_toolbox.utils import average_duplicated_centroids, \
    get_cluster_centroids_from_partition

import argparse


parser = argparse.ArgumentParser()

parser.add_argument("-output_dir", "--output_dir",
                    help="directory where the outputs will be stored",
                    type=str)
parser.add_argument("-label_name", "--label_name",
                    help="name of category to be segmented",
                    type=str)
parser.add_argument("-partition", "--partition",
                    help="name of category to be segmented",
                    type=str)
parser.add_argument("-xdim", "--xdim",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-ydim", "--ydim",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-zdim", "--zdim",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-overlap", "--overlap",
                    help="name of category to be segmented",
                    type=int, default=12)
parser.add_argument("-class_number", "--class_number",
                    help="class number associated to motifs list",
                    type=int)
parser.add_argument("-min_cluster_size", "--min_cluster_size",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-max_cluster_size", "--max_cluster_size",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-particle_radius", "--particle_radius",
                    help="name of category to be segmented",
                    type=int)


args = parser.parse_args()
output_dir = args.output_dir
label_name = args.label_name
partition = args.partition
xdim = args.xdim
ydim = args.ydim
zdim = args.zdim
overlap = args.overlap
class_number = args.class_number
min_cluster_size = args.min_cluster_size
max_cluster_size = args.max_cluster_size
particle_radius = args.particle_radius

output_shape = (zdim, ydim, xdim)
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
    motl_coords=full_centroids_list,
    min_peak_distance=particle_radius)

motl_name = "motl_" + str(len(unique_centroids)) + ".csv"
motl_file_name = join(output_dir, motl_name)
print("min point[2]", np.min([point[2] for point in unique_centroids]))
print("min point[1]", np.min([point[1] for point in unique_centroids]))
print("min point[0]", np.min([point[0] for point in unique_centroids]))

print("max point[2]", np.max([point[2] for point in unique_centroids]))
print("max point[1]", np.max([point[1] for point in unique_centroids]))
print("max point[0]", np.max([point[0] for point in unique_centroids]))

motive_list_df = build_tom_motive_list(
    list_of_peak_coordinates=unique_centroids, in_tom_format=False)
motive_list_df.to_csv(motl_file_name, index=False, header=False)
print("Motive list saved in", motl_file_name)
