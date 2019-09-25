import os

import datetime
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import tensorboardX as tb

from src.python.networks.unet_new import UNetEncoder3D
from src.python.networks.utils import generate_train_val_loaders
from src.python.networks.io import get_device
from src.python.filereaders.h5 import load_classification_training_set


class LinearTail(nn.Module):
    def __init__(self, in_features: int, out_features: int, p: float = 0.2):
        super().__init__()

        self.linear_tail = nn.Sequential(
            nn.Linear(in_features=in_features,
                      #           out_features=in_features // 2),
                      # nn.ReLU(),
                      # # nn.Dropout(p),
                      # nn.Linear(in_features=in_features // 2,
                      out_features=out_features),
        )

    def forward(self, input_tensor):
        x = self.linear_tail(input_tensor)
        return x


class Classifier3D(nn.Module):
    def get_linear_size(self):
        encoder_output_channels = 2 ** self.depth * self.initial_features
        encoder_output_side = self.volume_side // (2 ** self.depth)
        print("encoder_output_side", encoder_output_side)
        print("encoder_output_channels", encoder_output_channels)
        linear_size = encoder_output_channels * encoder_output_side ** 3
        return linear_size

    def __init__(self, in_channels: int, volume_side: int, output_classes: int,
                 depth: int, initial_features: int,
                 final_activation: nn.Softmax(dim=1)):
        super().__init__()
        # self.final_activation = final_activation
        self.final_activation = final_activation
        self.depth = depth
        self.initial_features = initial_features
        number_binnings = 1
        self.volume_side = volume_side // 2 ** number_binnings

        self.encoder = UNetEncoder3D(in_channels=in_channels, depth=depth,
                                     initial_features=initial_features)

        self.linear_size = self.get_linear_size()
        self.linear_tail = LinearTail(in_features=self.linear_size,
                                      out_features=output_classes)

        # print(self.linear_tail)

    def train_epoch(self, loader, loss_fn, optimizer, epoch, device,
                    print_stats=1, tb_logger=None, log_images=False):

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
            sl = self.volume_side // 2
            if batch_id % print_stats == 0:
                print('[{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                    (batch_id + 1) * len(x),
                    len(loader.dataset), 100. * (batch_id + 1) / len(loader),
                    loss.item()))
            if tb_logger is not None:
                tb_logger.add_scalar('train/loss', loss.item(), epoch)

                if log_images:
                    # write images
                    x_im = vutils.make_grid(x[:, :, sl, :, :],
                                            normalize=True,
                                            scale_each=True)

                    tb_logger.add_image('train/x', x_im, epoch)

        loss_avg = loss_cum / cnt
        print('Average Train Loss: {:.6f}'.format(loss_avg))
        return loss_avg

    def test_epoch(self, loader, loss_fn, epoch, device,
                   print_stats=1, tb_logger=None, log_images=False):

        # make sure network is on the gpu and in training mode
        self.to(device)
        self.eval()

        # keep track of the average loss during the epoch
        loss_cum = 0.0
        cnt = 0
        # start epoch
        for batch_id, (x, y) in enumerate(loader):
            # move input and target to the active device (either cpu or gpu)
            x, y = x.to(device), y.to(device)

            # apply model, calculate loss and run backwards pass
            prediction = self.forward(x)
            loss = loss_fn(prediction, y)
            loss_cum += loss.data.cpu().numpy()
            cnt += 1

            # To image:
            sl = self.volume_side // 2
            if batch_id % print_stats == 0:
                print('[{}/{} ({:.0f}%)]\tTest Loss: {:.6f}'.format(
                    (batch_id + 1) * len(x),
                    len(loader.dataset), 100. * (batch_id + 1) / len(loader),
                    loss.item()))
            if tb_logger is not None:
                tb_logger.add_scalar('test/loss', loss.item(), epoch)

                if log_images:
                    # write images
                    x_im = vutils.make_grid(x[:, :, sl, :, :],
                                            normalize=True,
                                            scale_each=True)

                    tb_logger.add_image('train/x', x_im, epoch)

        loss_avg = loss_cum / cnt
        print('Average Test Loss: {:.6f}'.format(loss_avg))
        return loss_avg

    def train_net(self, train_loader, test_loader, optimizer, loss_fn, device,
                  scheduler=None, epochs=100, test_freq=1,
                  print_stats=1, log_dir=None, log_images=False,
                  write_images_freq=1):

        # log everything if necessary
        if log_dir is not None:
            writer = tb.SummaryWriter(log_dir=log_dir)
        else:
            writer = None

        test_loss_min = np.inf
        for epoch in range(epochs):

            print('Epoch %5d/%5d' % (epoch + 1, epochs))

            if log_images:
                log_images = epoch % write_images_freq == 0
            else:
                log_images = False
            # train the model for one epoch
            self.train_epoch(loader=train_loader,
                             optimizer=optimizer, loss_fn=loss_fn,
                             epoch=epoch, device=device,
                             print_stats=print_stats, tb_logger=writer,
                             log_images=log_images)

            # adjust learning rate if necessary
            if scheduler is not None:
                scheduler.step(epoch=epoch)

                # and keep track of the learning rate
                writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]),
                                  epoch)

            # test the model for one epoch is necessary
            if epoch % test_freq == 0:
                test_loss = self.test_epoch(loader=test_loader,
                                            loss_fn=loss_fn,
                                            epoch=epoch, device=device,
                                            tb_logger=writer,
                                            log_images=log_images)

                #     # and save model if lower test loss is found
                #     if test_loss < test_loss_min:
                #         test_loss_min = test_loss
                #         torch.save(self,
                #                    os.path.join(log_dir, 'best_checkpoint.pytorch'))
                #         # ToDo add add test_epoch
                # # save model every epoch
                # torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

        writer.close()

    def forward(self, input_tensor):
        x = nn.AvgPool3d(kernel_size=2, stride=2)(input_tensor)
        _, x = self.encoder.forward(input_tensor=x)
        x = x.view(-1, self.linear_size)
        x = self.linear_tail.forward(x)
        x = self.final_activation(x)
        return x


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-train_log_dir", "--train_log_dir",
                    type=str,
                    default="/g/scb2/zaugg/trueba/3d-cnn/logs_classification3D/")
parser.add_argument("-data_path", "--data_path",
                    type=str,
                    default="/scratch/trueba/3Dclassifier/liang_data/training_data/200/training_set.h5")
parser.add_argument("-semantic_classes", "--semantic_classes",
                    type=str, default='70S,50S')
parser.add_argument("-model_initial_name", "--model_initial_name",
                    type=str, default='test_CrossEntropyLoss_')
parser.add_argument("-volume_side", "--volume_side",
                    type=int, default=64)
parser.add_argument("-spl", "--spl",
                    type=float, default=0.7)
parser.add_argument("-epochs", "--epochs",
                    type=int, default=40)
parser.add_argument("-output_classes", "--output_classes",
                    type=int, default=2)
parser.add_argument("-depth", "--depth",
                    type=int, default=3)
parser.add_argument("-initial_features", "--initial_features",
                    type=int, default=4)
args = parser.parse_args()

train_log_dir = args.train_log_dir

data_path = args.data_path

volume_side = args.volume_side

spl = args.spl

epochs = args.epochs

output_classes = args.output_classes

depth = args.depth

initial_features = args.initial_features

semantic_classes = args.semantic_classes
semantic_classes = semantic_classes.split(',')
model_initial_name = args.model_initial_name

# volume_side = 64
# spl = 0.7  # training set percentage
# epochs = 40  # 40  # 100
# depth = 3  # 2
# initial_features = 8  # 4
# output_classes = 2
#
# train_log_dir = "/g/scb2/zaugg/trueba/3d-cnn/logs_classification3D/"
# data_path = "/scratch/trueba/3Dclassifier/liang_data/training_data/200/training_set.h5"
# # data_path = "/scratch/trueba/3Dclassifier/liang_data/training_data/200/toy_training_set.h5"
# semantic_classes = ['70S', '50S']
# model_initial_name = "Test_CroSSEntLoss_"

# loss = DiceCoefficientLoss()
# final_activation = nn.Sigmoid()
# TODO idea: to be checked! already impllemented
# add a mean pooling at the beginning to reduce the size of images nn.AvgPool3d

# This does not work well!!!!!:
# loss = nn.NLLLoss()#F.nll_loss  # nn.NLLLoss()  #  Neither works that well
# final_activation = nn.LogSoftmax(dim=1) #dim=0  dimensions: B,Ch

loss = nn.CrossEntropyLoss()  # F.nll_loss  # nn.NLLLoss()  #  Neither works that well
final_activation = nn.ReLU()  # dim=0  dimensions: B,Ch

############## Load Dataset
# df = pd.read_csv(dataset_table)
# df['tomo_name'] = df['tomo_name'].astype(str)
# path_to_output_h5 = df.loc[df['tomo_name'] == tomo_name, 'train_partition']

volumes_raw, volumes_label = load_classification_training_set(semantic_classes,
                                                              data_path)

print("volumes_raw.shape, volumes_label.shape", volumes_raw.shape,
      volumes_label.shape)

now = datetime.datetime.now()
date = str(now.year) + "." + str(now.month) + "." + str(now.day)
time = str(now.hour) + "h." + str(now.minute) + "min."

net_name = model_initial_name + "D_" + str(depth) + "_IF_" + str(
    initial_features) + "_epochs_" + str(epochs) + "_" + date + "_" + time + "_"

log_dir = os.path.join(train_log_dir, net_name)

src_split = int(spl * volumes_raw.shape[0])
src_train_loader, src_val_loader = \
    generate_train_val_loaders(volumes_raw, volumes_label, src_split,
                               batch_size=2,
                               shuffle=True)

print(len(src_train_loader), len(src_val_loader))
############## Declare model:

classifier = Classifier3D(in_channels=1, volume_side=volume_side,
                          output_classes=output_classes, depth=depth,
                          initial_features=initial_features,
                          final_activation=final_activation)
device = get_device()
# loss_class = F.nll_loss

optimizer = optim.Adam(classifier.parameters())

# test train epoch:
# logger = tb.SummaryWriter(log_dir)
# classifier.train_epoch(loader=src_train_loader, loss_fn=loss,
#                        optimizer=optimizer, epoch=epochs, device=device,
#                        tb_logger=logger, log_images=False)

# test train network
classifier.train_net(train_loader=src_train_loader, test_loader=src_val_loader,
                     optimizer=optimizer, loss_fn=loss,
                     device=device, scheduler=None, epochs=epochs, test_freq=1,
                     print_stats=1, log_dir=log_dir, log_images=False,
                     write_images_freq=1)

# conf = {'in_channels': 1, 'final_activation': nn.ReLU(), 'depth': depth,
#         'initial_features': initial_features, 'volume_side': 64,
#         'output_classes': output_classes}
#
# import torch
#
# subtomo = torch.from_numpy(volumes_raw[:1, ...])
#
# transformed = nn.AvgPool3d(kernel_size=2, stride=2)(subtomo)
# print(transformed.shape)
#
# import matplotlib.pyplot as plt
#
# plt.imshow(transformed[0, 0, 16, :, :])
# plt.show()

# target = torch.from_numpy(volumes_label[:1, ...])
#
# print("target", target)
# ###
# model = Classifier3D(**conf)
# model.eval()
# prediction = model.forward(subtomo)
# print(prediction)
# print("softmax (p0, p1)=", nn.Softmax(dim=1)(prediction))
# print("-log(p1) =", loss(F.log_softmax(prediction), target).item())
#
#
# classifier.eval()
# output = classifier.forward(subtomo)
# print("LogSoftmax prediction", output)
# print("loss", loss(prediction, target).item())
# print("nll loss", loss(output, target).item())
