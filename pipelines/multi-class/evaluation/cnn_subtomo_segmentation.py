import torch
from src.python.pytorch_cnn.classes.unet import UNet
from src.python.pytorch_cnn.io import get_device
from src.python.filewriters.h5 import segment_and_write

# data to provide by user:
data_path = "/scratch/trueba/3d-cnn/training_data/multi-class/004/G_sigma1/train_and_test_partitions/partition_training.h5"
model_path = "/g/scb2/zaugg/trueba/3d-cnn/models/multi-class/w_1_64_1200_250_ribo_fas_memb_ribos_corrected_fas_memb_D_4_IF_8.pkl"
output_classes = 4
confs = {'final_activation': None,
         'depth': 4,
         'initial_features': 8,
         "out_channels": output_classes}
model = UNet(**confs)
label_name = "D_4_IF_8_w_1_64_1200_250"

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
