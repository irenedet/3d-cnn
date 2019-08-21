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
# from pytorch_cnn.classes.cnnets import UNet_4, UNet_6, UNet_7, UNet, UNet_deep
from networks.unet import UNet
from networks.loss import DiceCoefficient, \
    DiceCoefficientLoss
from networks.visualizers import TensorBoard
from networks.routines import train_float, validate_float
from networks.io import get_device

print("*************************************")
print("The cnn_training.py script is running")
print("*************************************")

# check if we have  a gpu
device = get_device()

training_data_path = \
    "/scratch/trueba/3d-cnn/training_data/TEST/data_aug.h5"
# '/scratch/trueba/3d-cnn/training_data/ribosomes/ribo_training_grid.h5'
# '/scratch/trueba/3d-cnn/training_data/training_data_side128_49examples.h5'
# '/scratch/trueba/3d-cnn/training_data/ribosomes/ribo_training_grid.h5'
# '/scratch/trueba/3d-cnn/training_data/ribosomes/ribo_training_grid.h5'
label_name = "ribosomes"
split = 170
print("The training data path is ", training_data_path)

raw_data, labels = h5.read_training_data(training_data_path,
                                         label_name=label_name)
print("labels shape", labels.shape)

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

train_data, train_labels, val_data, val_labels, data_order = \
    split_dataset(preprocessed_data, labels, split)

# print(data_order)

# wrap into datasets
train_set = du.TensorDataset(torch.from_numpy(train_data),
                             torch.from_numpy(train_labels))
val_set = du.TensorDataset(torch.from_numpy(val_data),
                           torch.from_numpy(val_labels))

# wrap into data-loader
train_loader = du.DataLoader(train_set, shuffle=False,  # we shuffle before
                             batch_size=5)
val_loader = du.DataLoader(val_set, batch_size=5)

for test_index in range(1):
    # train the neural network
    # net = UNet_deep(1, 1, final_activation=nn.Sigmoid())
    # net = UNet_7(1, 1, final_activation=nn.Sigmoid())
    # net = UNet(1, 1, final_activation=nn.Sigmoid())
    # net = UNet_test(1, 1, final_activation=nn.Sigmoid())

    # Multi class:
    # net = UNetGN(in_channel=1, out_channel=13,
    #              final_activation=nn.LogSoftmax(dim=1))


    net_confs = [
        {'depth': 2, 'initial_features': 8, 'final_activation': nn.Sigmoid()},
        ]
    is_best = True
    validation_loss = 10
    for conf in net_confs:
        net = UNet(**conf)
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
                     '_UNet_data_aug_sigmoid' + \
                     "d_" + str(conf['depth']) + \
                     "_ifeat_" + str(conf['initial_features']) + "_"
        model_name_pkl = model_name + ".pkl"
        log_dir = join('deepUNet_logs/', model_name)
        model_path = join("./models/", model_name_pkl)
        logger = TensorBoard(log_dir=log_dir, log_image_interval=1)
        print("The neural network training is now starting")
        n_epochs = 30
        for epoch in range(n_epochs):
            # apply training for one epoch
            train_float(net, train_loader, optimizer=optimizer,
                        loss_function=loss,
                        epoch=epoch, device=device, log_interval=1,
                        tb_logger=logger)
            step = epoch * len(train_loader.dataset)
            # run validation after training epoch
            current_validation_loss = validate_float(net, val_loader, loss,
                                                     metric,
                                                     device=device, step=step,
                                                     tb_logger=logger)

            if current_validation_loss < validation_loss:
                torch.save(net.state_dict(), model_path)
                validation_loss = current_validation_loss
            else:
                print("this model was not the best")

        model_name_txt = model_name + ".txt"
        data_txt_path = join("./models", model_name_txt)
        with open(data_txt_path, 'w') as txt_file:
            txt_file.write("training_data_path = " + training_data_path)
            txt_file.write("\n")
            txt_file.write("label_name=" + label_name)
            txt_file.write("\n")
            txt_file.write("split = " + str(split))
            txt_file.write("\n")
            txt_file.write("model_name = " + model_name_pkl)
            txt_file.write("\n")
            txt_file.write("conf = " + str(conf))
            txt_file.write("\n")
            txt_file.write("data_order = ")
            txt_file.write(str(data_order))
            txt_file.write("\n")

print("We have finished the training!")
