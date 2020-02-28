import torch

from file_actions.writers.h5 import segment_and_write
from networks.io import get_device
from networks.unet import UNet

# data to provide by user:
data_path = "/scratch/trueba/3d-cnn/training_data/multi-class/004/G_sigma1/train_and_test_partitions/partition_training.h5"
model_path = "/g/scb2/zaugg/trueba/3d-cnn/models/multi-class/ribo_corr_fas_memb_ribos_corrected_fas_memb_D_3_IF_8.pkl"
output_classes = 4
confs = {'final_activation': None,
         'depth': 3,
         'initial_features': 8,
         "out_channels": output_classes}
model = UNet(**confs)
label_name = "ribos_corrected_fas_memb"

# data_path = "/scratch/trueba/shrec/0_real_masks/training_sets/all_particles_foreground_training.h5"
# model_path = "/g/scb2/zaugg/trueba/3d-cnn/models/multi-class/Unet_all_parts_foreground_0all_particles_D_2_IF_16.pkl"
# model = UNet(1, 2, final_activation=None, depth=2, initial_features=16)
# label_name = "all_particles"

# data_path = "/scratch/trueba/3d-cnn/TEST/004_in_subtomos_128side_with_overlap.h5"
# model_path = "/g/scb2/zaugg/trueba/3d-cnn/models/0_lay_6_len_128_32_DiceLoss_ELUactiv_2ndtry.pkl"
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
