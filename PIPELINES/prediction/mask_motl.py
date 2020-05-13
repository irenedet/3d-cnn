import argparse
import os
from os import makedirs
from os.path import join

import numpy as np
import pandas as pd
import yaml

from constants.dataset_tables import ModelsTableHeader, DatasetTableHeader
from file_actions.readers.motl import read_motl_from_csv
from file_actions.readers.tomograms import load_tomogram
from file_actions.writers.csv import motl_writer
from networks.utils import build_prediction_output_dir

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
dataset_table = config['dataset_table']

ModelsHeader = ModelsTableHeader()
models_df = pd.read_csv(models_table,
                        dtype={ModelsHeader.model_name: str,
                               ModelsHeader.semantic_classes: str})
model_df = models_df[models_df[ModelsHeader.model_name] == model_name]
semantic_names = model_df.iloc[0]['segmentation_names'].split(',')
semantic_class = semantic_names[class_number]
ignore_border_thickness = config['motl_parameters'][
    'ignore_border_thickness']
filtering_mask = config['motl_parameters']['filtering_mask']

DTHeader = DatasetTableHeader(filtering_mask=filtering_mask)
df = pd.read_csv(dataset_table)
df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)

for tomo_name in tomo_list:
    output_dir = config['reconstruction']['reconstruction_path']
    output_dir = build_prediction_output_dir(base_output_dir=output_dir,
                                             label_name=label_name,
                                             model_name=model_name,
                                             tomo_name=tomo_name,
                                             semantic_class=semantic_class)

    motls_in_dir = [file for file in os.listdir(output_dir) if 'motl_' in file]
    assert len(motls_in_dir) == 1, "only one motive list can be filtered."
    csv_motl = os.path.join(output_dir, motls_in_dir[0])
    tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
    z_shift = int(tomo_df.iloc[0][DTHeader.z_shift])
    x_dim = int(tomo_df.iloc[0][DTHeader.x_dim])
    y_dim = int(tomo_df.iloc[0][DTHeader.y_dim])
    z_dim = int(tomo_df.iloc[0][DTHeader.z_dim])
    filtering_mask_path = tomo_df.iloc[0][DTHeader.filtering_mask]
    print("filtering mask file =", filtering_mask_path)
    conserved_points_dir = join(output_dir, "in_" + filtering_mask)
    discarded_points_dir = join(output_dir, "outside_" + filtering_mask)
    makedirs(conserved_points_dir, exist_ok=True)
    makedirs(discarded_points_dir, exist_ok=True)

    motl_predicted = read_motl_from_csv(path_to_csv_motl=csv_motl)
    motl_values = [row[0] for row in motl_predicted]
    predicted_coordinates = [np.array([row[7], row[8], row[9]]) for row in
                             motl_predicted]

    if isinstance(filtering_mask_path, float):
        print("Filtering mask file does not exist. "
              "All points will be conserved for the analysis.")
        conserved_values = motl_values
        conserved_points = predicted_coordinates
        discarded_values = []
        discarded_points = []
        motl_writer(path_to_output_folder=conserved_points_dir,
                    list_of_peak_scores=conserved_values,
                    list_of_peak_coords=conserved_points,
                    in_tom_format=True)
        motl_writer(path_to_output_folder=discarded_points_dir,
                    list_of_peak_scores=discarded_values,
                    list_of_peak_coords=discarded_points,
                    in_tom_format=True)
    else:
        filtering_mask_indicator = load_tomogram(
            path_to_dataset=filtering_mask_path)
        mask_z, mask_y, mask_x = filtering_mask_indicator.shape

        conserved_points = []
        conserved_values = []
        discarded_points = []
        discarded_values = []
        for value, point in zip(motl_values, predicted_coordinates):
            point = [int(entry) for entry in point]
            x, y, z = point
            if np.min([mask_x - x, mask_y - y, mask_z - z]) > 0 and np.min(
                    [x, y, z]) >= 0:
                if filtering_mask_indicator[z - z_shift, y, x] == 1 and np.min(
                        [x, y, x_dim - x, y_dim - y]) > ignore_border_thickness:
                    conserved_values += [value]
                    conserved_points += [point]
                else:
                    discarded_points += [point]
                    discarded_values += [value]

        motl_writer(path_to_output_folder=conserved_points_dir,
                    list_of_peak_scores=conserved_values,
                    list_of_peak_coords=conserved_points,
                    in_tom_format=True)
        motl_writer(path_to_output_folder=discarded_points_dir,
                    list_of_peak_scores=discarded_values,
                    list_of_peak_coords=discarded_points,
                    in_tom_format=True)
