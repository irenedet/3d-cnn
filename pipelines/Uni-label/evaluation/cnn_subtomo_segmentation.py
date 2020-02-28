import argparse

import torch

from networks.io import get_device
from networks.unet import UNet
from file_actions.writers.h5 import segment_and_write

parser = argparse.ArgumentParser()
parser.add_argument("-data_path", "--data_path",
                    help="path to partition tomogram in h5 format",
                    type=str)
parser.add_argument("-model", "--model_path",
                    help="path to model",
                    type=str)
parser.add_argument("-out_classes", "--output_classes",
                    help="number of output classes",
                    type=int)
parser.add_argument("-depth", "--depth",
                    help="depth of the unet",
                    type=int)
parser.add_argument("-init_feat", "--initial_features",
                    help="initial_features",
                    type=int)
parser.add_argument("-label", "--label_name",
                    help="name of category to be segmented",
                    type=str)


args = parser.parse_args()
data_path = args.data_path
model_path = args.model_path
output_classes = args.output_classes
depth = args.depth
initial_features = args.initial_features
label_name = args.label_name


# data to provide by user:
# data_path = "/scratch/trueba/3d-cnn/training_data/dice-multi-class/004/G_sigma1/train_and_test_partitions/partition_training.h5"
# model_path = "/g/scb2/zaugg/trueba/3d-cnn/models/dice_multi_label/w_1_1_1_ribos_corrected_fas_memb_D_2_IF_8.pkl"
# output_classes = 3


confs = {'final_activation': None,
         'depth': depth,
         'initial_features': initial_features,
         "out_channels": output_classes}
model = UNet(**confs)
# label_name = "D_2_IF_8_w_1_1_1"

device = get_device()
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.eval()

# data to segment
# data_dir = "/scratch/trueba/3d-cnn/TEST/"
# data_file = "004_in_subtomos_128side_with_overlap.h5"
# data_path = join(data_dir, data_file)

# save the segmented data in the same data file
segment_and_write(data_path, model, label_name=label_name)
print("The script has finished!")
