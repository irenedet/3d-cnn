import argparse
from distutils.util import strtobool

import pandas as pd
import torch
import torch.nn as nn

from src.python.networks.io import get_device
from src.python.networks.unet import UNet
from src.python.filewriters.h5 import segment_and_write

parser = argparse.ArgumentParser()

parser.add_argument("-model", "--path_to_model",
                    help="path to already trained net in .pkl format",
                    type=str)
parser.add_argument("-label", "--label_name",
                    help="name of category to be segmented",
                    type=str)
parser.add_argument("-tomo_name", "--tomo_name",
                    help="name of tomogram in format sessionname/datasetnumber",
                    type=str)
parser.add_argument("-dataset_table", "--dataset_table",
                    help="path to dataset table",
                    type=str)
parser.add_argument("-init_feat", "--initial_features",
                    help="Initial number of filters to apply in the UNet",
                    type=int)
parser.add_argument("-depth", "--unet_depth",
                    help="Depth of the UNet",
                    type=int)
parser.add_argument("-out_classes", "--output_classes",
                    help="Integer indicating number of classes to segment",
                    type=int)
parser.add_argument("-new_loader", "--new_loader",
                    help="Boolena indicating if loader is updated",
                    type=str,
                    default=False)

args = parser.parse_args()
path_to_model = args.path_to_model
label_name = args.label_name
dataset_table = args.dataset_table
tomo_name = args.tomo_name
init_feat = args.initial_features
depth = args.unet_depth
output_classes = args.output_classes
new_loader = strtobool(args.new_loader)


df = pd.read_csv(dataset_table)
df['tomo_name'] = df['tomo_name'].astype(str)
tomo_df = df[df['tomo_name'] == tomo_name]
data_path = tomo_df.iloc[0]['test_partition']

conf = {'final_activation': nn.ELU(), 'depth': depth,
        'initial_features': init_feat, 'out_channels': output_classes}

model = UNet(**conf)
device = get_device()
if new_loader:
    checkpoint = torch.load(path_to_model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(torch.load(path_to_model, map_location=device))
model = model.eval()

# save the segmented data in the same data file
segment_and_write(data_path=data_path, model=model, label_name=label_name)
print("The script has finished!")
