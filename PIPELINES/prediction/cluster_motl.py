import argparse
import os
from os import listdir

import pandas as pd
import torch
import torch.nn as nn
import yaml

from constants.dataset_tables import ModelsTableHeader, DatasetTableHeader
from file_actions.readers.tomograms import load_tomogram
from file_actions.writers.csv import build_tom_motive_list
from file_actions.writers.h5 import write_dataset_hdf
from networks.utils import build_prediction_output_dir
from tomogram_utils.coordinates_toolbox.clustering import get_cluster_centroids

parser = argparse.ArgumentParser()
parser.add_argument("-yaml_file", "--yaml_file", help="yaml_file", type=str)
parser.add_argument("-tomos_set", "--tomos_set",
                    help="tomos set name to be used for training", type=int)
args = parser.parse_args()
yaml_file = args.yaml_file
config = yaml.safe_load(open(yaml_file))
tomos_set = args.tomos_set
tomo_list = config['tomos_sets'][tomos_set]['test_list']

class_number = config['reconstruction']['class_number']
unet_hyperparameters = config['unet_hyperparameters']

if 'model_name' in config['tomos_sets'][tomos_set].keys():
    model_name = config['tomos_sets'][tomos_set]['model_name']
else:
    model_name = unet_hyperparameters['model_name']

label_name = unet_hyperparameters['label_name']
models_table = unet_hyperparameters['models_table']
motl_parameters = config['motl_parameters']
min_cluster_size = motl_parameters['min_cluster_size']
max_cluster_size = motl_parameters['max_cluster_size']
overlap = config['overlap']
box_shape = config['box_shape']

if isinstance(box_shape, int):
    box_shape = [box_shape, box_shape, box_shape]
else:
    box_shape = tuple([int(elem) for elem in reversed(box_shape)])

assert len(box_shape) == 3, "Invalid box_shape"

segmentation_label = label_name + "_" + model_name

ModelsHeader = ModelsTableHeader()
models_df = pd.read_csv(models_table,
                        dtype={ModelsHeader.model_name: str,
                               ModelsHeader.semantic_classes: str})
model_df = models_df[models_df[ModelsHeader.model_name] == model_name]
semantic_names = model_df.iloc[0][ModelsHeader.semantic_classes].split(',')
semantic_class = semantic_names[class_number]

dataset_table = config['dataset_table']
partition_name = config["partition_name"]
DTHeader = DatasetTableHeader()

for tomo_name in tomo_list:
    print("Processing tomo", tomo_name)
    output_dir = config['reconstruction']['reconstruction_path']
    output_dir = build_prediction_output_dir(base_output_dir=output_dir,
                                             label_name=label_name,
                                             model_name=model_name,
                                             tomo_name=tomo_name,
                                             semantic_class=semantic_class)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(dataset_table)
    df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)
    tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
    x_dim = int(tomo_df.iloc[0][DTHeader.x_dim])
    y_dim = int(tomo_df.iloc[0][DTHeader.y_dim])
    z_dim = int(tomo_df.iloc[0][DTHeader.z_dim])
    output_shape = (z_dim, y_dim, x_dim)
    calculate_motl = True
    for file in listdir(output_dir):
        if "motl" in file:
            print("Motive list already exists:", file)
            calculate_motl = False
    if calculate_motl:
        output_path = os.path.join(output_dir, "prediction.hdf")
        assert os.path.isfile(output_path)
        prediction_dataset = load_tomogram(path_to_dataset=output_path)

        sigmoid = nn.Sigmoid()
        prediction_dataset = sigmoid(
            torch.from_numpy(prediction_dataset).float())
        prediction_dataset = 1 * (prediction_dataset > 0.5).float()
        prediction_dataset = prediction_dataset.numpy()
        prediction_dataset.astype(int)
        clustering_labels, centroids_list, cluster_size_list = \
            get_cluster_centroids(dataset=prediction_dataset,
                                  min_cluster_size=min_cluster_size,
                                  max_cluster_size=max_cluster_size,
                                  connectivity=1)

        clusters_output_path = os.path.join(output_dir, "clusters.hdf")

        if not os.path.isfile(clusters_output_path):
            write_dataset_hdf(output_path=clusters_output_path,
                              tomo_data=clustering_labels)

        os.makedirs(output_dir, exist_ok=True)

        motl_name = "motl_" + str(len(centroids_list)) + ".csv"
        motl_file_name = os.path.join(output_dir, motl_name)

        if len(centroids_list) > 0:
            motive_list_df = build_tom_motive_list(
                list_of_peak_coordinates=centroids_list,
                list_of_peak_scores=cluster_size_list, in_tom_format=False)
            motive_list_df.to_csv(motl_file_name, index=False, header=False)
            print("Motive list saved in", motl_file_name)
        else:
            print("Saving empty list!")
            motive_list_df = pd.DataFrame({})
            motive_list_df.to_csv(motl_file_name, index=False, header=False)
