import argparse

import pandas as pd
import torch
import yaml

from constants.dataset_tables import ModelsTableHeader, DatasetTableHeader
from file_actions.writers.h5 import segment_and_write
from networks.io import get_device
from networks.unet import UNet3D

parser = argparse.ArgumentParser()
parser.add_argument("-yaml_file", "--yaml_file", help="yaml_file", type=str)
parser.add_argument("-tomo_name", "--tomo_name", help="tomo_name", type=str)
args = parser.parse_args()
yaml_file = args.yaml_file
tomo_name = args.tomo_name

config = yaml.safe_load(open(yaml_file))

unet_hyperparameters = config['unet_hyperparameters']
model_name = unet_hyperparameters['model_name']
label_name = unet_hyperparameters['label_name']
label_name = label_name + "_" + model_name
models_table = unet_hyperparameters['models_table']
dataset_table = config['dataset_table']
output_dir = config['output_dir']
overlap = config['overlap']
box_shape = config['box_shape']
if isinstance(box_shape, int):
    box_shape = [box_shape, box_shape, box_shape]
else:
    box_shape = tuple([int(elem) for elem in reversed(box_shape)])

assert len(box_shape) == 3, "Invalid box_shape"

DTHeader = DatasetTableHeader()
df = pd.read_csv(dataset_table, dtype={DTHeader.tomo_name: str})
tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
data_path = tomo_df.iloc[0][DTHeader.partition_name]
print("test_partition", data_path)

ModelsHeader = ModelsTableHeader()
models_df = pd.read_csv(models_table, dtype={ModelsHeader.model_name: str})
model_df = models_df[models_df[ModelsHeader.model_name] == model_name]
print(model_df)
path_to_model = model_df.iloc[0][ModelsHeader.model_path]
init_feat = model_df.iloc[0][ModelsHeader.initial_features]
depth = model_df.iloc[0][ModelsHeader.depth]
output_classes = model_df.iloc[0][ModelsHeader.output_classes]
BN = model_df.iloc[0][ModelsHeader.batch_normalization].astype(bool)
encoder_dropout = model_df.iloc[0][ModelsHeader.encoder_dropout]
decoder_dropout = model_df.iloc[0][ModelsHeader.decoder_dropout]

net_conf = {'final_activation': None,
            'depth': depth,
            'initial_features': init_feat,
            "out_channels": output_classes,
            "BN": BN,
            "encoder_dropout": encoder_dropout,
            "decoder_dropout": decoder_dropout}

model = UNet3D(**net_conf)

device = get_device()
checkpoint = torch.load(path_to_model, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.eval()

# save the segmented data in the same data file
segment_and_write(data_path=data_path, model=model, label_name=label_name)
print("The script has finished!")
