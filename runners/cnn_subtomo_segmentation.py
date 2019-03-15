from os.path import join
import torch
from src.python.pytorch_cnn.classes.cnnets import UNet_6
from src.python.pytorch_cnn.classes.unet_new import UNet
import torch.nn as nn
from src.python.pytorch_cnn.io import get_device
from src.python.filewriters.h5 import segment_and_write

# data to provide by user:
data_path = "/scratch/trueba/shrec/0_test/ALL_particles_training.h5"
model_path = "/g/scb2/zaugg/trueba/3d-cnn/models/0_large_sph_ALL_particles_depth_2_ini_feat_8.pkl"
model = UNet(1, 13, final_activation=None, depth=2, initial_features=8)
label_name = "large_sph_no_activation"



# data_path = "/scratch/trueba/3d-cnn/TEST/004_in_subtomos_128side_with_overlap.h5"
# model_path = "/g/scb2/zaugg/trueba/3d-cnn/models/0_lay_6_len_128_32_DiceLoss_ELUactiv_2ndtry.pkl"
# model = UNet_6(1, 1, final_activation=nn.ELU()) #ToDo automate this step
#model = UNet(**kwargs??)

# load pre trained model
# model_dir = "/g/scb2/zaugg/trueba/3d-cnn/models"
# model_name_pkl = '0_lay_6_len_128_32_DiceLoss_ELUactiv_2ndtry.pkl'
# model_path = join(model_dir, model_name_pkl)
# Here the user can change the final activation a modo:
# model = UNet_6(1, 1, final_activation=nn.ELU())
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
