from os.path import join
import numpy as np
import h5py
import torch
from src.python.pytorch_cnn.classes.cnnets import UNet_6
import torch.nn as nn
import matplotlib.pyplot as plt


def read_data_from_h5(data_path: str) -> np.array:
    data = []
    with h5py.File(data_path, 'r') as f:
        for subtomo_name in list(f['volumes/subtomos']):
            subtomo_h5_internal_path = join('volumes/subtomos', subtomo_name)
            data += [f[subtomo_h5_internal_path][:]]
    return np.array(data)


def write_segmented_data(data_path: str, output: np.array,
                         label_name: str) -> np.array:
    with h5py.File(data_path, 'a') as f:
        for subtomo_indx, subtomo_name in enumerate(
                list(f['volumes/subtomos'])):
            subtomo_label_path = join('volumes/labels/',
                                      label_name)
            subtomo_h5_internal_path = join(subtomo_label_path,
                                            subtomo_name)
            f[subtomo_h5_internal_path] = output[subtomo_indx, :, :, :, :]
    return np.array(data)


# load pre trained model
model_dir = "/g/scb2/zaugg/trueba/3d-cnn/models"
model_name_pkl = '0_lay_6_len_128_32_DiceLoss_ELUactiv_2ndtry.pkl'
model_path = join(model_dir, model_name_pkl)
model = UNet_6(1, 1, final_activation=nn.ELU())
model.load_state_dict(torch.load(model_path))
# model.load_state_dict(torch.load(model_path, map_location='cpu'))
model = model.eval()

# load data
data_dir = "/scratch/trueba/3d-cnn/evaluating_data/"
data_file = "subtomo_data_path.h5"  # ""tomo004_in_subtomos_128side.h5"
data_path = join(data_dir, data_file)
data = read_data_from_h5(data_path)
data = np.array(data)[:, None]

# save data
output = model(torch.from_numpy(data))
output = output.detach().numpy()
write_segmented_data(data_path, output, 'ribosomes')
