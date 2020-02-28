from os.path import join
import numpy as np

from file_actions.readers.tomograms import load_tomogram
from coordinates_toolbox.clustering import get_cluster_centroids
from file_actions.writers.h5 import write_dataset_hdf


input_path = "/struct/mahamid/Irene/yeast/ED/181119/030/memb/tomosegresult_thr_43.hdf"
output_dir = "/struct/mahamid/Irene/yeast/ED/181119/030/memb"
motl = load_tomogram(path_to_dataset=input_path)

clustering_labels, centroids_list, cluster_size_list = \
    get_cluster_centroids(dataset=motl,
                          min_cluster_size=300000,
                          max_cluster_size=np.inf,
                          connectivity=2)

clusters_output_path = join(output_dir, "clusters.hdf")
write_dataset_hdf(output_path=clusters_output_path, tomo_data=clustering_labels)