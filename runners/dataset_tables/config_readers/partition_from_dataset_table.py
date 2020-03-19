import argparse
import os
from distutils.util import strtobool
from os import makedirs

import numpy as np
import pandas as pd
import yaml

from file_actions.readers.tomograms import load_tomogram
from tomogram_utils.volume_actions.actions import \
    partition_raw_intersecting_mask
from tomogram_utils.volume_actions.actions import partition_tomogram

parser = argparse.ArgumentParser()
parser.add_argument("-yaml_file", "--yaml_file", help="yaml_file", type=str)
args = parser.parse_args()
yaml_file = args.yaml_file
config = yaml.safe_load(open(yaml_file))
tomo_names = config['tomo_names']
dataset_table = config['datasets_description']['dataset_table']
global_output_dir = config['partition_parameters']['global_output_dir']

box_shape = config['unet_hyperparameters']['box_side']
overlap = config['partition_parameters']['box_overlap']
write_on_table = strtobool(
    config['partition_parameters']['write_partition_on_table'])
if isinstance(box_shape, int):
    box_shape = [box_shape, box_shape, box_shape]
else:
    box_shape = tuple([int(elem) for elem in reversed(box_shape)])


assert len(box_shape) == 3, "Invalid box_shape"

for tomo_name in tomo_names:
    output_dir = os.path.join(global_output_dir, tomo_name)
    output_partition_path = os.path.join(output_dir, "partition.h5")
    makedirs(output_dir, exist_ok=True)
    dataset_table_df = pd.read_csv(dataset_table,  dtype={'tomo_name': str})
    tomo_df = dataset_table_df[dataset_table_df['tomo_name'] == tomo_name]
    path_to_raw = tomo_df.iloc[0]['eman2_filetered_tomo']
    path_to_lamella = tomo_df.iloc[0]['lamella_file']

    raw_dataset = load_tomogram(path_to_dataset=path_to_raw)

    if isinstance(path_to_lamella, float):
        print("No lamella file available.")
        partition_tomogram(dataset=raw_dataset,
                           output_h5_file_path=output_partition_path,
                           subtomo_shape=box_shape, overlap=overlap)
    else:
        path_to_lamella = tomo_df.iloc[0]['lamella_file']
        lamella_mask = load_tomogram(path_to_dataset=path_to_lamella)
        lamella_shape = lamella_mask.shape
        dataset_shape = raw_dataset.shape

        minimum_shape = [np.min([data_dim, lamella_dim]) for data_dim, lamella_dim
                         in zip(dataset_shape, lamella_shape)]
        minz, miny, minx = minimum_shape

        lamella_mask = lamella_mask[:minz, :miny, :minx]
        raw_dataset = raw_dataset[:minz, :miny, :minx]

        partition_raw_intersecting_mask(dataset=raw_dataset,
                                        mask_dataset=lamella_mask,
                                        output_h5_file_path=output_partition_path,
                                        subtomo_shape=box_shape,
                                        overlap=overlap)

    if write_on_table:
        dataset_table_df.loc[
            dataset_table_df['tomo_name'] == tomo_name, 'test_partition'] = \
            output_partition_path
        dataset_table_df.to_csv(dataset_table, index=False)
