from os.path import join
import numpy as np
# import h5py
import torch
from src.python.pytorch_cnn.classes.cnnets import UNet_6
import torch.nn as nn
from src.python.pytorch_cnn.io import get_device
# from src.python.naming import h5_internal_paths
from src.python.filereaders.h5 import read_raw_data_from_h5
from src.python.filewriters.h5 import write_segmented_data

# def read_raw_data_from_h5(data_path: str) -> np.array:
#     data = []
#     with h5py.File(data_path, 'r') as f:
#         for subtomo_name in list(f[h5_internal_paths.RAW_SUBTOMOGRAMS]):
#             subtomo_h5_internal_path = join(h5_internal_paths.RAW_SUBTOMOGRAMS,
#                                             subtomo_name)
#             data += [f[subtomo_h5_internal_path][:]]
#     return np.array(data)


# def write_segmented_data(data_path: str, output_segmentation: np.array,
#                          label_name: str) -> np.array:
#     with h5py.File(data_path, 'a') as f:
#         for subtomo_indx, subtomo_name in enumerate(
#                 list(f[h5_internal_paths.RAW_SUBTOMOGRAMS])):
#             segmented_subtomo_path = join(
#                 h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS,
#                 label_name)
#             subtomo_h5_internal_path = join(segmented_subtomo_path,
#                                             subtomo_name)
#             f[subtomo_h5_internal_path] = output_segmentation[subtomo_indx, :,
#                                           :, :, :]



# load pre trained model
model_dir = "/g/scb2/zaugg/trueba/3d-cnn/models"
model_name_pkl = '0_lay_6_len_128_32_DiceLoss_ELUactiv_2ndtry.pkl'
model_path = join(model_dir, model_name_pkl)
# Here the user can change the final activation a modo:
model = UNet_6(1, 1, final_activation=nn.ELU())
device = get_device()
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.eval()

# load data
data_dir = "/scratch/trueba/3d-cnn/TEST/"
data_file = "004_in_subtomos_128side.h5"
data_path = join(data_dir, data_file)
data = read_raw_data_from_h5(data_path)
data = np.array(data)[:, None]

# save data
output = model(torch.from_numpy(data))
output = output.detach().numpy()

write_output = True
if write_output:
    write_segmented_data(data_path, output, label_name="ribosomes")
else:
    print("The output was not written.")
    print("To write, change the value of write_output to True")
print("The script has finished!")
