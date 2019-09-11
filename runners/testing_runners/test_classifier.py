import os
import datetime
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from src.python.networks.unet_new import UNetEncoder3D
from src.python.networks.utils import generate_train_val_loaders, \
    data_loader
from src.python.networks.io import get_device
import torchvision.utils as vutils


class Classifier3D(nn.Module):
    def __init__(self, in_channels: int, output_classes: int, depth: int,
                 initial_features: int, final_activation: nn.Softmax(dim=1)):
        super().__init__()
        # self.final_activation = final_activation
        self.final_activation = final_activation
        encoder_output_channels = 2 ** (depth - 1) * initial_features
        self.encoder = UNetEncoder3D(in_channels=in_channels, depth=depth,
                                     initial_features=initial_features)
        self.linear_tail = nn.Sequential(
            nn.Linear(in_features=encoder_output_channels,
                      out_features=encoder_output_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=encoder_output_channels // 2,
                      out_features=output_classes),
            self.final_activation)
        # self.classifier3D = nn.ModuleList(
        #     [self.encoder, self.linear_tail, self.final_activation])

    # def train_epoche(self, x, y):
    #     loss = nn.functional.nll_loss(prediction, y)
    def train_epoch(self, loader, loss_fn, optimizer, epoch, device,
                    print_stats=1, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.to(device)
        self.train()

        # keep track of the average loss during the epoch
        loss_cum = 0.0
        cnt = 0

        # start epoch
        for batch_id, (x, y) in enumerate(loader):
            # move input and target to the active device (either cpu or gpu)
            x, y = x.to(device), y.to(device)

            # zero the gradients for this iteration
            optimizer.zero_grad()

            # apply model, calculate loss and run backwards pass
            prediction = self.forward(x)
            loss = loss_fn(prediction, y)
            loss.backward()
            optimizer.step()
            loss_cum += loss.data.cpu().numpy()
            cnt += 1

            # To image:
            _, _, depth_dim, _, _ = prediction.shape
            sl = depth_dim // 2
            if batch_id % print_stats == 0:
                print('[{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                    (batch_id + 1) * len(x),
                    len(loader.dataset), 100. * (batch_id + 1) / len(loader),
                    loss.item()))
            if writer is not None:
                writer.add_scalar('train/loss-seg', loss.item(), epoch)

                if write_images:
                    # write images
                    x_im = vutils.make_grid(x[:, :, sl, :, :],
                                            normalize=True,
                                            scale_each=True)

                    writer.add_image('train/x', x_im, epoch)
                    writer.add_image('train/y', y, epoch)
                    writer.add_image('train/y_pred', prediction, epoch)

        loss_avg = loss_cum / cnt
        print('Average Train Loss: {:.6f}'.format(loss_avg))
        return loss_avg

    def forward(self, input_tensor):
        _, x = self.encoder.forward(input_tensor=input_tensor)
        x = self.linear_tail.forward(x)
        x = self.final_activation(x)
        return x


final_activation = nn.LogSoftmax(dim=1)  # dim=1 channel dimension: B,Ch,H,W,T

classifier = Classifier3D(in_channels=1, output_classes=3, depth=2,
                          initial_features=8, final_activation=final_activation)
device = get_device()
loss_class = F.nll_loss()
optimizer = optim.Adam(classifier.parameters())

src_data = ""
spl = 0.8  # 0.8  # training set percentage
epochs = 100  # 100
depth = 2
initial_features = 8
train_log_dir = "/g/scb2/zaugg/trueba/3d-cnn/logs_classification3D/"

now = datetime.datetime.now()
date = str(now.month) + "." + str(now.day)
time = str(now.hour) + "." + str(now.minute) + "." + str(now.second)

net_name = date + "_" + time + "_" + "D_" + str(depth) + "_IF_" + str(
    initial_features)
log_dir = os.path.join(train_log_dir, net_name)

src_raw, src_label = data_loader(data_path=src_data, semantic_class='ribo')
# trg_raw, trg_label = data_loader(data_path=trg_data, semantic_class='ribo')

src_split = int(spl * src_raw.shape[0])
src_train_loader, src_val_loader = \
    generate_train_val_loaders(src_raw, src_label, src_split, batch_size=2,
                               shuffle=True)

""" Test for encoder-decoder: working for depth=2! """
# ##### encoder-decoder net:
train_log_dir = "/g/scb2/zaugg/trueba/3d-cnn/classifyer3D_logs/"


# src_train_loader, src_val_loader = \
#     generate_train_val_loaders(src_raw, src_raw, split, batch_size=5,
#                                shuffle=True)
#
