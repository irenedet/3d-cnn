import argparse
import os
from os.path import join

import pandas as pd
import yaml

from constants import h5_internal_paths
from constants.dataset_tables import ModelsTableHeader, DatasetTableHeader
from file_actions.writers.h5 import \
    assemble_tomo_from_subtomos
from networks.utils import build_prediction_output_dir

parser = argparse.ArgumentParser()
parser.add_argument("-yaml_file", "--yaml_file", help="yaml_file", type=str)
parser.add_argument("-tomo_name", "--tomo_name", help="tomo_name in dataset "
                                                      "table", type=str)
args = parser.parse_args()
tomo_name = args.tomo_name
yaml_file = args.yaml_file
config = yaml.safe_load(open(yaml_file))

unet_hyperparameters = config['unet_hyperparameters']
model_name = unet_hyperparameters['model_name']
label_name = unet_hyperparameters['label_name']
segmentation_label = label_name + "_" + model_name
overlap = config['overlap']
box_shape = config['box_shape']
if isinstance(box_shape, int):
    box_shape = [box_shape, box_shape, box_shape]
else:
    box_shape = tuple([int(elem) for elem in reversed(box_shape)])

assert len(box_shape) == 3, "Invalid box_shape"

class_number = config['reconstruction']['class_number']


ModelsHeader = ModelsTableHeader()
models_table = unet_hyperparameters['models_table']
models_df = pd.read_csv(models_table,
                        dtype={ModelsHeader.model_name: str,
                               ModelsHeader.semantic_classes: str})
model_df = models_df[models_df[ModelsHeader.model_name] == model_name]
semantic_names = model_df.iloc[0]['segmentation_names'].split(',')
semantic_class = semantic_names[class_number]

output_dir = config['reconstruction']['reconstruction_path']
output_dir = build_prediction_output_dir(base_output_dir=output_dir,
                                         label_name=label_name,
                                         model_name=model_name,
                                         tomo_name=tomo_name,
                                         semantic_class=semantic_class)
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "prediction.hdf")

dataset_table = config['dataset_table']
partition_name = config["partition_name"]
DTHeader = DatasetTableHeader(partition_name=partition_name)

df = pd.read_csv(dataset_table)
df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)
tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
x_dim = int(tomo_df.iloc[0][DTHeader.x_dim])
y_dim = int(tomo_df.iloc[0][DTHeader.y_dim])
z_dim = int(tomo_df.iloc[0][DTHeader.z_dim])
output_shape = (z_dim, y_dim, x_dim)
data_partition = tomo_df.iloc[0][DTHeader.partition_name]

subtomos_internal_path = join(
    h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS,
    segmentation_label)

assemble_tomo_from_subtomos(output_path=output_path,
                            partition_file_path=data_partition,
                            output_shape=output_shape, subtomo_shape=box_shape,
                            subtomos_internal_path=subtomos_internal_path,
                            class_number=class_number, overlap=overlap,
                            reconstruction_type="prediction")
