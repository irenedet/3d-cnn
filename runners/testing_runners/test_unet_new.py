from os.path import join

import h5py
import numpy as np
import torch
import torch.utils.data as du

from networks.io import get_device
from networks.unet_new import UNet3D

dummy_data = "/scratch/trueba/3d-cnn/training_data/dice-multi-class/180426_004/G_sigma1/train_and_test_partitions/test_partition.h5"

src_raw = list()
src_label = list()
trg_raw = list()
trg_label = list()
N = 1
with h5py.File(dummy_data, 'r') as f:
    source_raw_names = list(f['volumes/raw'])[:N]
    source_label_names = list(f['volumes/labels/ribo'])[:N]
    # target_raw_names = list(f['volumes/raw'])[N:2*N]
    for source_vol_name in source_raw_names:
        src_raw_path = join('volumes/raw', source_vol_name)
        src_label_path = join('volumes/labels/ribo', source_vol_name)
        src_raw += [f[src_raw_path][:]]
        src_label += [f[src_label_path][:]]

# datasets
src_raw, src_label = np.array(src_raw)[None, :], np.array(src_label)[None, :]

src_raw, src_label = src_raw.transpose((1, 0, 2, 3, 4)), src_label.transpose(
    (1, 0, 2, 3, 4))

# print(src_raw.shape)

device = get_device()

train_data = src_raw
train_labels = src_label
val_data = src_raw
val_labels = src_label

train_set = du.TensorDataset(torch.from_numpy(train_data),
                             torch.from_numpy(train_labels))
val_set = du.TensorDataset(torch.from_numpy(val_data),
                           torch.from_numpy(val_labels))

train_loader = du.DataLoader(train_set, shuffle=False, batch_size=5)
val_loader = du.DataLoader(val_set, batch_size=5)

# unet = UNet3D(in_channels=1, out_channels=1, depth=1, initial_features=2,
#               final_activation=nn.Sigmoid(), skip_connections=False, elu=False)
# optimizer = optim.Adam(unet.parameters())
# train_log_dir = "/home/papalotl/"
# log_dir = os.path.join(train_log_dir, 'finetuning')
#
# loader = train_loader
# weight = [1]
# weight_tensor = torch.tensor(weight).to(device)
# loss = DiceCoefficientLoss_multilabel(weights=weight)
# loss = loss.to(device)
# loss_function = loss
# unet.train_net(train_loader=loader, test_loader=loader, device=device,
#                log_dir=log_dir, loss_fn=loss, epochs=3)

path_to_model = "/g/scb2/zaugg/trueba/3d-cnn/ynet3D_logs/100_epoches_src_004_trg_005_D_4_IF_8/best_checkpoint.pytorch"
confs = {'in_channels': 1,
         "out_channels": 1,
         "depth": 4,
         "initial_features": 8,
         "skip_connections": True,
         # "final_activation": nn.Sigmoid(),
         # "elu": False
         }

from networks.utils import load_unet_model

unet, optimizer, epoch, loss = load_unet_model(path_to_model=path_to_model,
                                               confs=confs,
                                               mode="eval", net=UNet3D)

unet(torch.from_numpy(val_data))
