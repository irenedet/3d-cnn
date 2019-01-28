#! /home/trueba/.conda/envs/mlcourse/bin/python3

import sys
from os.path import join

# import os
# import re


# TODO!
# This only works when calling the submission script from
# submission_scripts/:

# nb_dir = os.path.split(os.getcwd())[0]
# print(nb_dir)
# pysrc = "/src/python"
# print(re.findall(r"(.+?3d-cnn)", nb_dir))
# project_path = re.findall(r"(.+?3d-cnn)", nb_dir)[0]
# py_src_path = project_path + pysrc

# Change these lines so that they work regardless of where I
# call the running script!

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
from image.filters import preprocess_data
from image.viewers import view_images_h5
from pytorch_cnn.classes.cnnets import UNet_4, UNet_6, UNet_7, UNet, UNet_deep
from pytorch_cnn.classes.loss import BCELoss, DiceCoefficient, \
    DiceCoefficientLoss
from pytorch_cnn.classes.visualizers import TensorBoard
from pytorch_cnn.classes.routines import train, validate
from pytorch_cnn.io import get_device

print("*************************************")
print("The cnn_training.py script is running")
print("*************************************")

# check if we have  a gpu
device = get_device()

training_data_path = \
    '/scratch/trueba/3d-cnn/training_data/training_data_side128_49examples.h5'
# '/scratch/trueba/3d-cnn/training_data/ribosomes/ribo_training_grid.h5'
# '/scratch/trueba/3d-cnn/training_data/ribosomes/ribo_training_grid.h5'


print("The training data path is ", training_data_path)

raw_data, labels = h5.read_training_data(training_data_path,
                                         label_name="ribosomes")

print("Initial unique labels", np.unique(labels))

view_data = False

if view_data:
    view_images_h5(data_path=training_data_path, img_range=(5, 10))
else:
    print(
        "The data viewer is deactivated, to activate it set view_data to True")

# Normalize data
preprocessed_data = preprocess_data(raw_data)

# add a channel dimension
preprocessed_data = np.array(preprocessed_data)[:, None]
labels = np.array(labels)[:, None]

train_data, train_labels, val_data, val_labels = \
    split_dataset(preprocessed_data, labels, 42-8)

# wrap into datasets
train_set = du.TensorDataset(torch.from_numpy(train_data),
                             torch.from_numpy(train_labels))
val_set = du.TensorDataset(torch.from_numpy(val_data),
                           torch.from_numpy(val_labels))

# wrap into data-loader
train_loader = du.DataLoader(train_set, shuffle=True, batch_size=5)
val_loader = du.DataLoader(val_set, batch_size=5)

for test_index in range(2):
    # train the neural network
    net = UNet_deep(1, 1, final_activation=nn.Sigmoid())
    # net = UNet_7(1, 1, final_activation=nn.Sigmoid())
    # net = UNet(1, 1, final_activation=nn.Sigmoid())
    # net = UNet_test(1, 1, final_activation=nn.Sigmoid())
    net = net.to(device)

    # built binary cross without weighting and adam optimizer
    # loss = BCELoss()
    loss = DiceCoefficientLoss()
    loss = loss.to(device)
    optimizer = optim.Adam(net.parameters())

    # build the dice coefficient metric
    metric = DiceCoefficient()
    metric = metric.to(device)

    # built tensorboard logger
    model_name = str(
        test_index) + \
                 'UNet_deep_128_side'
    log_dir = join('deepUNet_logs/', model_name)
    logger = TensorBoard(log_dir=log_dir,
                         log_image_interval=1)  # log every image
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
    model_name_pkl = model_name + ".pkl"
    model_path = join("./models/", model_name_pkl)
    torch.save(net.state_dict(), model_path)

print("We have finished the training!")
