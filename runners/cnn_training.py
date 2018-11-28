#! /home/trueba/.conda/envs/mlcourse/bin/python3

import sys

py_src_path = "/g/scb2/zaugg/trueba/3d-cnn/src/python"
sys.path.append(py_src_path)
runners_path = "/g/scb2/zaugg/trueba/3d-cnn/runners"
sys.path.append(runners_path)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as du

from datasets.actions import split_dataset
from filereaders import h5
from image.filters import normalize_training_data
from image.viewers import view_images_h5
from pytorch_cnn.classes.cnnets import UNet_bis
from pytorch_cnn.classes.loss import BCELoss, DiceCoefficient
from pytorch_cnn.classes.visualizers import TensorBoard
from pytorch_cnn.classes.routines import train, validate

print("*************************************")
print("The cnn_training.py script is running")
print("*************************************")

# check if we have  a gpu
if torch.cuda.is_available():
    print("GPU is available")
    device = torch.device("cuda")
else:
    print("GPU is not available")
    device = torch.device("cpu")

training_data_path = \
    '/scratch/trueba/3d-cnn/training_data/training_data_side128_49examples.h5'
print("The training data path is ", training_data_path)

raw_data, labels = h5.read_training_data(training_data_path)

print("Initial unique labels", np.unique(labels))

view_data = False

if view_data:
    view_images_h5(data_path=training_data_path, img_range=(5, 10))
else:
    print(
        "The data viewer is deactivated, to activate it set view_data to True")
normalized_data = normalize_training_data(raw_data)

# add a channel dimension
normalized_data = np.array(normalized_data)[:, None]
labels = np.array(labels)[:, None]

train_data, train_labels, val_data, val_labels = \
    split_dataset(normalized_data, labels, 39)

# wrap into datasets
train_set = du.TensorDataset(torch.from_numpy(train_data),
                             torch.from_numpy(train_labels))
val_set = du.TensorDataset(torch.from_numpy(val_data),
                           torch.from_numpy(val_labels))

# wrap into data-loader
train_loader = du.DataLoader(train_set, shuffle=True, batch_size=5)
val_loader = du.DataLoader(val_set, batch_size=5)

for test_index in range(5):
    # train the neural network
    net = UNet_bis(1, 1, final_activation=nn.Sigmoid())
    net = net.to(device)

    # built binary cross without weighting and adam optimizer
    loss = BCELoss()
    loss = loss.to(device)
    optimizer = optim.Adam(net.parameters())

    # build the dice coefficient metric
    metric = DiceCoefficient()
    metric = metric.to(device)

    # built tensorboard logger
    log_dir = 'logs/' + str(
        test_index) + 'TEST_7_layers_side_length_128_39from42__examples'
    logger = TensorBoard(log_dir, 20)  # log every 20th image

    # train for 25 epochs
    # after the  you can inspect the
    # predictions in the tensorboard
    print("The neural network training is now starting")
    n_epochs = 30
    for epoch in range(n_epochs):
        # apply training for one epoch
        train(net, train_loader, optimizer=optimizer, loss_function=loss,
              epoch=epoch, device=device, log_interval=1, tb_logger=logger)
        step = epoch * len(train_loader.dataset)
        # run validation after training epoch
        validate(net, val_loader, loss, metric, device=device, step=step,
                 tb_logger=logger)

print("goodbye!")
