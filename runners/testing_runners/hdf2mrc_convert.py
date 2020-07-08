from file_actions.readers.tomograms import load_tomogram
from tomogram_utils.coordinates_toolbox.clustering import get_clusters_within_size_range

from file_actions.writers.datasets import write_mrc_dataset
import os
import numpy as np
from tqdm import tqdm
#
# tomos = [
#     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out2/180426_004_pred_pred_polished.mrc",
#     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out2/180426_005_pred_pred_polished.mrc",
#     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out2/180426_021_pred_pred_polished.mrc",
#     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out2/180426_024_pred_pred_polished.mrc",
#     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out2/180426_026_pred_polished.mrc",
#     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out2/180426_027_pred_polished.mrc",
#     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out2/180426_028_pred_polished.mrc",
#     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out2/180426_029_pred_polished.mrc",
#     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out2/180426_030_pred_polished.mrc",
#     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out2/180426_034_pred_polished.mrc",
#     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out2/180426_037_pred_polished.mrc",
#     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out2/180426_041_pred_polished.mrc",
#     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out2/180426_043_pred_polished.mrc",
#     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out2/180426_045_pred_polished.mrc",
#     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out2/180711_003_pred_polished.mrc",
#     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out2/180711_004_pred_polished.mrc",
#     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out2/180711_005_pred_polished.mrc",
#     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out2/180711_018_pred_polished.mrc",
#     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out2/180713_027_pred_polished.mrc",
#     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out2/181119_002_pred_pred_polished.mrc",
#     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out2/181119_030_pred_pred_polished.mrc",
#     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out2/181126_002_pred_pred_polished.mrc",
#     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out2/181126_012_pred_pred_polished.mrc",
#     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out2/181126_025_pred_pred_polished.mrc",
# ]
#
# # tomos = [
# #     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out/180426_004_pred_pred_polished_clusters_bin.mrc",
# #     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out/180426_005_pred_pred_polished_clusters_bin.mrc",
# #     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out/180426_021_pred_pred_polished_clusters_bin.mrc",
# #     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out/180426_024_pred_pred_polished_clusters_bin.mrc",
# #     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out/180426_026_pred_polished_clusters_bin.mrc",
# #     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out/180426_027_pred_polished_clusters_bin.mrc",
# #     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out/180426_028_pred_polished_clusters_bin.mrc",
# #     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out/180426_029_pred_polished_clusters_bin.mrc",
# #     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out/180426_030_pred_polished_clusters_bin.mrc",
# #     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out/180426_034_pred_polished_clusters_bin.mrc",
# #     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out/180426_037_pred_polished_clusters_bin.mrc",
# #     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out/180426_041_pred_polished_clusters_bin.mrc",
# #     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out/180426_043_pred_polished_clusters_bin.mrc",
# #     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out/180426_045_pred_polished_clusters_bin.mrc",
# #     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out/180711_003_pred_polished_clusters_bin.mrc",
# #     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out/180711_004_pred_polished_clusters_bin.mrc",
# #     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out/180711_005_pred_polished_clusters_bin.mrc",
# #     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out/180711_018_pred_polished_clusters_bin.mrc",
# #     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out/180713_027_pred_polished_clusters_bin.mrc",
# #     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out/181119_002_pred_pred_polished_clusters_bin.mrc",
# #     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out/181119_030_pred_pred_polished_clusters_bin.mrc",
# #     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out/181126_002_pred_pred_polished_clusters_bin.mrc",
# #     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out/181126_012_pred_pred_polished_clusters_bin.mrc",
# #     "/struct/mahamid/Irene/cytoplasm_detection_cluster/memb/out/181126_025_pred_pred_polished_clusters_bin.mrc",
# # ]
#
# lamellas = [
#     "/struct/mahamid/Irene/yeast/healthy/180426/004/lamellamask0.03_bin.hdf",
#     "/struct/mahamid/Irene/yeast/healthy/180426/005/lamellamask0.03_bin.hdf",
#     "/struct/mahamid/Irene/yeast/healthy/180426/021/lamellamask0.03_bin.hdf",
#     "/struct/mahamid/Irene/yeast/healthy/180426/024/lamellamask0.03_bin.hdf",
#     "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/026/TM/026_lamellamask.em",
#     "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/027/TM/027_lamellamask.em",
#     "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/028/TM/028_lamellamask.em",
#     "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/029/TM/029_lamellamask.em",
#     "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/030/TM/030_lamellamask.em",
#     "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/034/TM/034_lamellamask.em",
#     "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/037/TM/037_lamellamask.em",
#     "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/041/TM/041_lamellamask.em",
#     "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/043/TM/043_lamellamask.em",
#     "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/045/TM/045_lamellamask.em",
#     "/struct/mahamid/Irene/yeast/healthy/180711/003/lamellamask.hdf",
#     "/struct/mahamid/Irene/yeast/healthy/180711/004/lamellamask.hdf",
#     "/struct/mahamid/Irene/yeast/healthy/180711/005/lamellamask.hdf",
#     "/struct/mahamid/Irene/yeast/healthy/180711/018/lamellamask.hdf",
#     "/struct/mahamid/Irene/yeast/healthy/180713/027/lamellamask.hdf",
#     "/struct/mahamid/Irene/yeast/ED/181119/002/lamellamask.hdf",
#     "/struct/mahamid/Irene/yeast/ED/181119/030/lamellamask.hdf",
#     "/struct/mahamid/Irene/yeast/ED/181126/002/lamellamask.hdf",
#     "/struct/mahamid/Irene/yeast/ED/181126/012/lamellamask.hdf",
#     "/struct/mahamid/Irene/yeast/ED/181126/025/lamellamask.hdf",
# ]
#
# min_cluster_size = 3000
# max_cluster_size = 5579232
# for tomo, lamella in zip(tomos, lamellas):
#     print(tomo)
#     output_dir = os.path.dirname(tomo)
#     output_name = os.path.basename(tomo)[:-4] + "_clustered_bin_masked.mrc"
#     output_path = os.path.join(output_dir, output_name)
#     tomo_array = load_tomogram(path_to_dataset=tomo)
#
#     labeled_clusters, labels_list_within_range, cluster_size_within_range = \
#         get_clusters_within_size_range(dataset=tomo_array, min_cluster_size=min_cluster_size,
#                                        max_cluster_size=max_cluster_size, connectivity=1)
#     selected_clusters = np.zeros(labeled_clusters.shape)
#     print("cluster_size_within_range =", cluster_size_within_range)
#     for index, size, label in zip(tqdm(range(len(cluster_size_within_range))),
#                                   cluster_size_within_range,
#                                   labels_list_within_range):
#         cluster_tmp = 1 * (labeled_clusters == label)
#         selected_clusters += cluster_tmp
#
#     lamella_array = load_tomogram(path_to_dataset=lamella)
#     if lamella_array.shape == selected_clusters.shape:
#         print(lamella_array.shape, selected_clusters.shape)
#         masked_array = selected_clusters * lamella_array
#
#     else:
#         sz, sy, sx = lamella_array.shape
#         print(lamella_array.shape, selected_clusters.shape)
#         new_lamella = np.zeros(selected_clusters.shape)
#         new_lamella[:sz, :sy, :sx] = lamella_array
#         masked_array = selected_clusters * new_lamella
#
#     write_mrc_dataset(mrc_path=output_path,
#                       array=masked_array)

file = "/struct/mahamid/Irene/test-3d-unet/out/180426/004/test_part_alex_filter.h5"

with h5py.File(file, "r") as f:
    print(len(list(f['volumes/predictions/membrane_model_new_training_part'])))
    for subtomo in subtomos[30:39]:
        r_path = os.path.join('volumes/raw', subtomo)
        l_path = os.path.join('volumes/predictions/membrane_model_new_training_part', subtomo)
        raws.append(f[r_path][:])
        labels.append(f[l_path][:])

