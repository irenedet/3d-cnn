import argparse
from distutils.util import strtobool
import yaml
import numpy as np
import pandas as pd
import torch

from file_actions.writers.h5 import segment_and_write
from networks.io import get_device
from networks.unet import UNet, UNet_BN, UNet_dropout

# parser = argparse.ArgumentParser()
#
# parser.add_argument("-model", "--path_to_model",
#                     help="path to already trained net in .pkl format",
#                     type=str)
# parser.add_argument("-label", "--label_name",
#                     help="name of category to be segmented",
#                     type=str)
# parser.add_argument("-tomo_name", "--tomo_name",
#                     help="name of tomogram in format sessionname/datasetnumber",
#                     type=str)
# parser.add_argument("-dataset_table", "--dataset_table",
#                     help="path to dataset table",
#                     type=str)
# parser.add_argument("-init_feat", "--initial_features",
#                     help="Initial number of filters to apply in the UNet",
#                     type=int)
# parser.add_argument("-depth", "--unet_depth",
#                     help="Depth of the UNet",
#                     type=int)
# parser.add_argument("-BN", "--Batch_Normalization",
#                     help="Batch_Normalization",
#                     type=str, default="False")
# parser.add_argument("-encoder_dropout", "--encoder_dropout",
#                     help="encoder_dropout",
#                     type=float, default=0)
# parser.add_argument("-decoder_dropout", "--decoder_dropout",
#                     help="decoder_dropout",
#                     type=float, default=0)
# parser.add_argument("-out_classes", "--output_classes",
#                     help="Integer indicating number of classes to segment",
#                     type=int)
# parser.add_argument("-new_loader", "--new_loader",
#                     help="Boolena indicating if loader is updated",
#                     type=str, default=False)
#
# args = parser.parse_args()

parser = argparse.ArgumentParser()
parser.add_argument("-yaml_file", "--yaml_file", help="yaml_file", type=str)
parser.add_argument("-tomo_name", "--tomo_name", help="tomo_name", type=str)
args = parser.parse_args()
yaml_file = args.yaml_file
tomo_name = args.tomo_name

config = yaml.safe_load(open(yaml_file))

unet_hyperparameters = config['unet_hyperparameters']
path_to_model = unet_hyperparameters['path_to_model']
label_name = unet_hyperparameters['label_name']
init_feat = unet_hyperparameters['init_feat']
depth = unet_hyperparameters['depth']
output_classes = unet_hyperparameters['output_classes']
new_loader = strtobool(unet_hyperparameters['new_loader'])
BN = strtobool(unet_hyperparameters['BN'])
encoder_dropout = unet_hyperparameters['encoder_dropout']
decoder_dropout = unet_hyperparameters['decoder_dropout']
print("BN = ", BN)
print("encoder_dropout, decoder_dropout", encoder_dropout, decoder_dropout)
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
    df = pd.read_csv(dataset_table,  dtype={'tomo_name': str})
    tomo_df = df[df['tomo_name'] == tomo_name]
    data_path = tomo_df.iloc[0]['test_partition']
    print("test_partition", data_path)
    conf = {'final_activation': None, 'depth': depth,
            'initial_features': init_feat, 'out_channels': output_classes}
    if not BN:
        model = UNet(**conf)
    else:
        model = UNet_BN(**conf)
    if np.max([encoder_dropout, decoder_dropout]) > 0:
        conf['encoder_dropout'] = encoder_dropout
        conf['decoder_dropout'] = decoder_dropout
        model = UNet_dropout(**conf)
    else:
        print("No dropout")

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
