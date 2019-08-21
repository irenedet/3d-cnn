import torch.nn as nn

from networks.io import get_device
from src.python.networks.utils import load_unet_model

model = "/g/scb2/zaugg/trueba/3d-cnn/mixed_models/0_UNET_8TOMOS_DATA_AUG__D2_IF4_elu_.pkl"
confs = {'depth': 2, 'initial_features': 4, 'elu': True,
         'final_activation': nn.Sigmoid()}
device = get_device()

model, optimizer, epoch, loss = load_unet_model(path_to_model=model,
                                                confs=confs, mode="train")
