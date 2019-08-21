import argparse

import torch
import torch.nn as nn

from networks.io import get_device
from networks.unet import UNet
from src.python.filewriters.h5 import segment_and_write

parser = argparse.ArgumentParser()

parser.add_argument("-model", "--path_to_model",
                    help="path to already trained net in .pkl format",
                    type=str)
parser.add_argument("-label", "--label_name",
                    help="name of category to be segmented",
                    type=str)
parser.add_argument("-outh5", "--output_h5_path",
                    help="file where the outputs will be stored",
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

args = parser.parse_args()
path_to_model = args.path_to_model
label_name = args.label_name
output_h5file = args.output_h5_path
init_feat = args.initial_features
depth = args.unet_depth
output_classes = args.output_classes

conf = {'final_activation': nn.ELU(), 'depth': depth,
        'initial_features': init_feat, 'out_channels': output_classes}

model = UNet(**conf)

device = get_device()
model.load_state_dict(torch.load(path_to_model, map_location=device))
model = model.eval()

# save the segmented data in the same data file
segment_and_write(data_path=output_h5file, model=model, label_name=label_name)
print("The script has finished!")
