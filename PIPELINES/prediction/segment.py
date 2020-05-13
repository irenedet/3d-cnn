import argparse

import pandas as pd
import torch
import yaml

from constants.dataset_tables import ModelsTableHeader, DatasetTableHeader
from file_actions.writers.h5 import segment_and_write
from networks.io import get_device
from networks.unet import UNet3D

# from networks.unet import UNet #, UNet_dropout

parser = argparse.ArgumentParser()
parser.add_argument("-yaml_file", "--yaml_file", help="yaml_file", type=str)
parser.add_argument("-tomos_set", "--tomos_set",
                    help="tomos set name to be used for training", type=int)

args = parser.parse_args()
yaml_file = args.yaml_file
config = yaml.safe_load(open(yaml_file))
tomos_set = args.tomos_set
tomo_list = config['tomos_sets'][tomos_set]['test_list']
unet_hyperparameters = config['unet_hyperparameters']

if 'model_name' in config['tomos_sets'][tomos_set].keys():
    model_name = config['tomos_sets'][tomos_set]['model_name']
else:
    model_name = unet_hyperparameters['model_name']

label_name = unet_hyperparameters['label_name']
models_table = unet_hyperparameters['models_table']
dataset_table = config['dataset_table']
overlap = config['overlap']
box_shape = config['box_shape']
test_partition = config['partition_name']
if isinstance(box_shape, int):
    box_shape = [box_shape, box_shape, box_shape]
else:
    box_shape = tuple([int(elem) for elem in reversed(box_shape)])

assert len(box_shape) == 3, "Invalid box_shape"
ModelsHeader = ModelsTableHeader()
models_df = pd.read_csv(models_table, dtype=ModelsHeader.dtype_dict)
model_df = models_df[models_df[ModelsHeader.model_name] == model_name]
if model_df.shape[0] > 1:
    model_df = model_df[model_df[ModelsHeader.label_name] == label_name]

print(model_df)
assert model_df.shape[0] == 1
path_to_model = model_df.iloc[0][ModelsHeader.model_path]
initial_features = model_df.iloc[0][ModelsHeader.initial_features]
depth = model_df.iloc[0][ModelsHeader.depth]
output_classes = model_df.iloc[0][ModelsHeader.output_classes]
BN = model_df.iloc[0][ModelsHeader.batch_normalization].astype(bool)
encoder_dropout = model_df.iloc[0][ModelsHeader.encoder_dropout]
decoder_dropout = model_df.iloc[0][ModelsHeader.decoder_dropout]
label_name = label_name + "_" + model_name

net_conf = {'final_activation': None, 'depth': depth,
            'initial_features': initial_features, "out_channels": output_classes,
            "BN": BN, "encoder_dropout": encoder_dropout,
            "decoder_dropout": decoder_dropout}

model = UNet3D(**net_conf)

# model = UNet(in_channels=1, out_channels=output_classes,
#              depth=depth, initial_features=initial_features,
#              final_activation=None)

device = get_device()
checkpoint = torch.load(path_to_model, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.eval()

DTHeader = DatasetTableHeader(partition_name=test_partition)
df = pd.read_csv(dataset_table, dtype={DTHeader.tomo_name: str})
for tomo_name in tomo_list:
    print("Segmenting tomo", tomo_name)
    tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
    data_path = tomo_df.iloc[0][DTHeader.partition_name]
    print("test_partition", data_path)
    segment_and_write(data_path=data_path, model=model, label_name=label_name)
    print("The script has finished!")
