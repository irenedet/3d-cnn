import argparse

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
parser.add_argument("-data_path", "--data_path",
                    help="file where the outputs will be stored",
                    type=str)
parser.add_argument("-init_feat", "--initial_features",
                    help="Initial number of filters to apply in the UNet",
                    type=int)
parser.add_argument("-depth", "--unet_depth",
                    help="Depth of the UNet",
                    type=int)
parser.add_argument("-new_loader", "--new_loader",
                    help="Boolean, if True, cnn loader is in the new format",
                    type=str)

args = parser.parse_args()
path_to_model = args.path_to_model
label_name = args.label_name
data_path = args.data_path
init_feat = args.initial_features
depth = args.unet_depth
new_loader = args.new_loader
if new_loader == 'True':
    new_loader = True
else:
    new_loader = False
print("new_loader =", new_loader)

conf = {'depth': depth, 'initial_features': init_feat}

model = UNet(**conf, final_activation=nn.ELU())
device = get_device()

if new_loader:
    checkpoint = torch.load(path_to_model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(torch.load(path_to_model, map_location=device))
model = model.eval()

# save the segmented data in the same data file
print("label_name = ", label_name)
segment_and_write(data_path=data_path, model=model, label_name=label_name)
print("The script has finished!")
