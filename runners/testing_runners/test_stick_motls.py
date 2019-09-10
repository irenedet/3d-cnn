import pandas as pd

motive_list_centroids = "/scratch/trueba/3d-cnn/cnn_evaluation/liang_dataset/clusters/test/peak_cluster_NO_DA_ribo_D_2_IF_8_pr_radius_30/full_dataset//246/class_0/combined_motl_25000.0_sph/small_clusters_motl.csv"
motive_list_peaks = "/scratch/trueba/3d-cnn/cnn_evaluation/liang_dataset/clusters/test/peak_cluster_NO_DA_ribo_D_2_IF_8_pr_radius_30/full_dataset//246/class_0/combined_motl_25000.0_sph/big_clusters_motl.csv"
output_motl_path = "/scratch/trueba/3d-cnn/cnn_evaluation/liang_dataset/clusters/test/peak_cluster_NO_DA_ribo_D_2_IF_8_pr_radius_30/full_dataset//246/class_0/combined_motl_25000.0_sph/combined_motl.csv"

motl_centroids = pd.read_csv(motive_list_centroids, header=None)
motl_peaks = pd.read_csv(motive_list_peaks, header=None)

combined_motl = motl_peaks.append(motl_centroids, sort=False)
combined_motl.to_csv(output_motl_path, header=False, index=False)


