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
parser.add_argument("-tomo_name", "--tomo_name", help="tomo_name in dataset "
                                                      "table", type=str)
args = parser.parse_args()
tomo_name = args.tomo_name
yaml_file = args.yaml_file
config = yaml.safe_load(open(yaml_file))

dataset_table = config['dataset_table']
output_dir = config['output_dir']
overlap = config['overlap']
write_on_table = config["write_on_table"]
partition_name = config["partition_name"]
processing_tomo = config["processing_tomo"]
box_shape = config["box_shape"]

if isinstance(box_shape, int):
    subtomogram_shape = (box_shape, box_shape, box_shape)
else:
    subtomogram_shape = tuple([int(elem) for elem in reversed(box_shape)])

os.path.join(output_dir, tomo_name)
os.makedirs(output_dir, exist_ok=True)

DTHeader = DatasetTableHeader(processing_tomo=processing_tomo,
                              partition_name=partition_name)

df = pd.read_csv(dataset_table)
df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)
tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
path_to_raw = tomo_df.iloc[0][DTHeader.processing_tomo]
path_to_lamella = tomo_df.iloc[0][DTHeader.lamella_file]

raw_dataset = load_tomogram(path_to_dataset=path_to_raw)
partition_path = os.path.join(output_dir, "grid_partition.h5")

if isinstance(path_to_lamella, float):
    print("No lamella file available.")
    partition_tomogram(dataset=raw_dataset, output_h5_file_path=partition_path,
                       subtomo_shape=subtomogram_shape, overlap=overlap)
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
                                    output_h5_file_path=partition_path,
                                    subtomo_shape=subtomogram_shape,
                                    overlap=overlap)

if write_on_table:
    df.loc[df[DTHeader.tomo_name] == tomo_name, DTHeader.partition_name] = \
        partition_path
    df.to_csv(dataset_table, index=False)
