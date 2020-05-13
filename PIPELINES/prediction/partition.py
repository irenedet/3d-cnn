import argparse
import os

import numpy as np
import pandas as pd
import yaml

from constants.dataset_tables import DatasetTableHeader
from file_actions.readers.tomograms import load_tomogram
from tomogram_utils.volume_actions.actions import \
    partition_raw_intersecting_mask
from tomogram_utils.volume_actions.actions import partition_tomogram

parser = argparse.ArgumentParser()
parser.add_argument("-yaml_file", "--yaml_file", help="yaml_file", type=str)
parser.add_argument("-tomos_set", "--tomos_set",
                    help="tomos set name to be used for training", type=int)
args = parser.parse_args()
yaml_file = args.yaml_file
config = yaml.safe_load(open(yaml_file))
tomos_set = args.tomos_set
tomo_list = config['tomos_sets'][tomos_set]['test_list']

dataset_table = config['dataset_table']
overlap = config['overlap']
write_on_table = config["write_on_table"]
partition_name = config["partition_name"]
processing_tomo = config["processing_tomo"]
box_shape = config["box_shape"]

if isinstance(box_shape, int):
    subtomogram_shape = (box_shape, box_shape, box_shape)
else:
    subtomogram_shape = tuple([int(elem) for elem in reversed(box_shape)])

DTHeader = DatasetTableHeader(processing_tomo=processing_tomo,
                              partition_name=partition_name)

df = pd.read_csv(dataset_table)
df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)

if write_on_table:
    for tomo_name in tomo_list:
        print("Partitioning tomo", tomo_name)
        output_dir = config['output_dir']
        output_dir_tomo = os.path.join(output_dir, tomo_name)
        os.makedirs(output_dir_tomo, exist_ok=True)
        partition_path = os.path.join(output_dir_tomo, partition_name + ".h5")
        print("output path:", partition_path)
        if os.path.isfile(partition_path):
            print("Partition exists already.")
        else:
            tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
            path_to_raw = tomo_df.iloc[0][DTHeader.processing_tomo]
            path_to_lamella = tomo_df.iloc[0][DTHeader.filtering_mask]
            raw_dataset = load_tomogram(path_to_dataset=path_to_raw)
            if isinstance(path_to_lamella, float):
                print("No filtering mask file available.")
                partition_tomogram(dataset=raw_dataset,
                                   output_h5_file_path=partition_path,
                                   subtomo_shape=subtomogram_shape,
                                   overlap=overlap)
            else:
                path_to_lamella = tomo_df.iloc[0][DTHeader.filtering_mask]
                lamella_mask = load_tomogram(path_to_dataset=path_to_lamella)

                lamella_shape = lamella_mask.shape
                dataset_shape = raw_dataset.shape

                minimum_shape = [np.min([data_dim, lamella_dim]) for
                                 data_dim, lamella_dim
                                 in zip(dataset_shape, lamella_shape)]
                minz, miny, minx = minimum_shape

                lamella_mask = lamella_mask[:minz, :miny, :minx]
                raw_dataset = raw_dataset[:minz, :miny, :minx]

                partition_raw_intersecting_mask(dataset=raw_dataset,
                                                mask_dataset=lamella_mask,
                                                output_h5_file_path=partition_path,
                                                subtomo_shape=subtomogram_shape,
                                                overlap=overlap)

        df.loc[df[
                   DTHeader.tomo_name] == tomo_name, DTHeader.partition_name] =\
            [partition_path]
        df.to_csv(dataset_table, index=False)
