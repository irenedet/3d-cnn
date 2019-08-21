import os
import datetime
import torch.nn as nn
import torch.optim as optim
import torch
from src.python.networks.ynet import YNet3D
from src.python.networks.unet_new import UNet3D
from src.python.networks.utils import generate_train_val_loaders, \
    data_loader
from src.python.networks.io import get_device
from src.python.networks.loss import DiceCoefficientLoss

src_data = "/struct/mahamid/Irene/yeast/healthy/180426/004/G_sigma1_non_sph/train_and_test_partitions/full_partition.h5"
trg_data = "/struct/mahamid/Irene/yeast/healthy/180426/005/G_sigma1_non_sph/train_and_test_partitions/full_partition.h5"
spl = 0.8  # 0.8  # training set percentage
epochs = 100  # 100
depth = 2
initial_features = 8
train_log_dir = "/g/scb2/zaugg/trueba/3d-cnn/ynet3D_logs/"

now = datetime.datetime.now()
date = str(now.month) + "." + str(now.day)
time = str(now.hour) + "." + str(now.minute) + "." + str(now.second)

net_name = date + "_" + time + "_" + "D_" + str(depth) + "_IF_" + str(
    initial_features)
log_dir = os.path.join(train_log_dir, net_name)

src_raw, src_label = data_loader(data_path=src_data, semantic_class='ribo')
trg_raw, trg_label = data_loader(data_path=trg_data, semantic_class='ribo')

src_split = int(spl * src_raw.shape[0])
src_train_loader, src_val_loader = \
    generate_train_val_loaders(src_raw, src_label, src_split, batch_size=2,
                               shuffle=True)

trg_split = int(spl * trg_raw.shape[0])
trg_train_loader, trg_val_loader = \
    generate_train_val_loaders(trg_raw, trg_label, trg_split, batch_size=2,
                               shuffle=True)

ynet = YNet3D(in_channels=1, out_channels=2, initial_features=initial_features,
              depth=depth, segm_final_activation=nn.Sigmoid())

device = get_device()
optimizer = optim.Adam(ynet.parameters())
loss_seg_fn = DiceCoefficientLoss()
loss_rec_fn = nn.MSELoss()

ynet.train_net(train_loader_source=src_train_loader,
               train_loader_target=trg_train_loader,
               test_loader_source=src_val_loader,
               test_loader_target=trg_val_loader,
               optimizer=optimizer, loss_seg_fn=loss_seg_fn,
               loss_rec_fn=loss_rec_fn, device=device, log_dir=log_dir,
               epochs=epochs)

""" Test for encoder-decoder: working for depth=2! """
# ##### encoder-decoder net:
# train_log_dir = "/g/scb2/zaugg/trueba/3d-cnn/encoder_decoder3D_logs/"
# src_train_loader, src_val_loader = \
#     generate_train_val_loaders(src_raw, src_raw, split, batch_size=5,
#                                shuffle=True)
#
# encoder_decoder = UNet3D(in_channels=1, out_channels=1, depth=depth,
#                          initial_features=initial_features,
#                          final_activation=None, skip_connections=False)
#
# device = get_device()
# optimizer = optim.Adam(encoder_decoder.parameters())
# loss_seg_fn = nn.MSELoss()
#
# encoder_decoder.train_net(train_loader=src_train_loader,
#                           test_loader=src_val_loader, device=device,
#                           log_dir=log_dir, loss_fn=loss_seg_fn, epochs=epochs)

""" Reloading a ynet model: """
# path_to_model = "/g/scb2/zaugg/trueba/3d-cnn/ynet3D_logs/100_epoches_src_004_trg_005_D_4_IF_8/best_checkpoint.pytorch"
# confs = {'in_channels': 1,
#          "out_channels": 1,
#          "depth": 4,
#          "initial_features": 8,
#          "segm_final_activation": nn.Sigmoid(),
#          # "elu": False
#          }
#
# from networks.utils import load_unet_model
# ynet = YNet3D(**confs)
#
# device = get_device()
# ynet.load_state_dict(torch.load(path_to_model, map_location=device))
# ynet = ynet.eval()
