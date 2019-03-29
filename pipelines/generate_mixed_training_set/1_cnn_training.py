#! /home/trueba/.conda/envs/mlcourse/bin/python3

import sys
from os.path import join

# import os
# import re


py_src_path = "/g/scb2/zaugg/trueba/3d-cnn/src/python"
sys.path.append(py_src_path)
runners_path = "/g/scb2/zaugg/trueba/3d-cnn/runners"
sys.path.append(runners_path)

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as du

from src.python.datasets.actions import split_dataset
from src.python.filereaders import h5
from src.python.image.filters import preprocess_data
from src.python.pytorch_cnn.classes.unet_new import UNet
from src.python.pytorch_cnn.classes.loss import BCELoss, DiceCoefficient, \
    DiceCoefficientLoss
from src.python.pytorch_cnn.classes.visualizers import TensorBoard
from src.python.pytorch_cnn.classes.routines import train_float, validate_float
from src.python.pytorch_cnn.io import get_device

print("*************************************")
print("The cnn_training.py script is running")
print("*************************************")


def get_testing_and_training_sets_from_partition(training_data_path: str,
                                                 split: tuple):
    print("The training data path is ", training_data_path)
    raw_data, labels = h5.read_training_data(training_data_path,
                                             label_name=label_name)
    print("Initial unique labels", np.unique(labels))

    # Normalize data
    preprocessed_data = preprocess_data(raw_data)

    # add a channel dimension
    preprocessed_data = np.array(preprocessed_data)[:, None]
    labels = np.array(labels)[:, None]

    train_data, train_labels, val_data, val_labels, data_order = \
        split_dataset(preprocessed_data, labels, split)
    return train_data, train_labels, val_data, val_labels, data_order


# check if we have  a gpu
device = get_device()

# Loading training and testing sets from different files
training_data_path = \
    "/scratch/trueba/3d-cnn/training_data/TEST/mixed_training/004/train_and_test_partitions/partition_training.h5"
label_name = "ribosomes"
# split = (110, 130)  # this works for 004 and 006
split = 110
print("Done loading training set from 004.")
train_data, train_labels, val_data, val_labels, _ = \
    get_testing_and_training_sets_from_partition(training_data_path, split)

training_data_path = \
    "/scratch/trueba/3d-cnn/training_data/TEST/mixed_training/005/train_and_test_partitions/partition_training.h5"

train_data_tmp, train_labels_tmp, val_data_tmp, val_labels_tmp, _ = \
    get_testing_and_training_sets_from_partition(training_data_path,
                                                 split=77)
                                                 # split=(77, 90))

print("Done loading training set from 005.")
train_data = np.concatenate((train_data, train_data_tmp), axis=0)
train_labels = np.concatenate((train_labels, train_labels_tmp), axis=0)
val_data = np.concatenate((val_data, val_data_tmp), axis=0)
val_labels = np.concatenate((val_labels, val_labels_tmp), axis=0)

training_data_path = \
    "/scratch/trueba/3d-cnn/training_data/TEST/mixed_training/006/train_and_test_partitions/partition_training.h5"

train_data_tmp, train_labels_tmp, val_data_tmp, val_labels_tmp, _ = \
    get_testing_and_training_sets_from_partition(training_data_path, split)
print("Done loading training set from 006.")
train_data = np.concatenate((train_data, train_data_tmp), axis=0)
train_labels = np.concatenate((train_labels, train_labels_tmp), axis=0)
val_data = np.concatenate((val_data, val_data_tmp), axis=0)
val_labels = np.concatenate((val_labels, val_labels_tmp), axis=0)

######################### Unet preparation##############################

# wrap into datasets
train_set = du.TensorDataset(torch.from_numpy(train_data),
                             torch.from_numpy(train_labels))
val_set = du.TensorDataset(torch.from_numpy(val_data),
                           torch.from_numpy(val_labels))

# wrap into data-loader
train_loader = du.DataLoader(train_set, shuffle=True,  # we shuffle before
                             batch_size=5)
val_loader = du.DataLoader(val_set, batch_size=5)

for test_index in range(2):

    net_confs = [{'depth': 5, 'initial_features': 4},
                 # {'depth': 5, 'initial_features': 8},
                 ]

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
        model_name = str(test_index) + '_UNET_shuffle_mixed_G_sigma1-5_' + \
                     "D_" + str(conf['depth']) + \
                     "_IF_" + str(conf['initial_features']) + "_"
        log_dir = join('mixed_logs/', model_name)
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
            validate_float(net, val_loader, loss, metric, device=device,
                           step=step,
                           tb_logger=logger)
        model_name_pkl = model_name + ".pkl"
        model_path = join("./mixed_models/", model_name_pkl)
        torch.save(net.state_dict(), model_path)

        model_name_txt = model_name + ".txt"
        data_txt_path = join("./mixed_models", model_name_txt)
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

print("We have finished the training!")