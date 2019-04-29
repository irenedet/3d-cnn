import torch
from src.python.pytorch_cnn.classes.unet import UNet
from src.python.pytorch_cnn.io import get_device
from src.python.filewriters.h5 import segment_and_write

# data to provide by user:
data_path = "/scratch/trueba/3d-cnn/training_data/dice-multi-class/004/G_sigma1/train_and_test_partitions/partition_training.h5"
model_path = "/g/scb2/zaugg/trueba/3d-cnn/models/dice_multi_label/0_w_1_1_1_ribo_fas_memb_D_2_IF_8.pkl"
output_classes = 3
confs = {'final_activation': None,
         'depth': 2,
         'initial_features': 8,
         "out_channels": output_classes}
model = UNet(**confs)
label_name = "D_2_IF_8_0_w_1_1_1"

device = get_device()
model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
model = model.eval()

# save the segmented data in the same data file
segment_and_write(data_path, model, label_name=label_name)
print("The script has finished!")
