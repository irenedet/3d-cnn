#! /home/trueba/.conda/envs/mlcourse/bin/python3

import sys
from os.path import join

from src.python.networks.utils import save_unet_model

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

from networks.utils import \
    get_testing_and_training_sets_from_partition
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
training_data_paths = [
    "/struct/mahamid/Irene/yeast/ribosomes/180426_004/G_sigma1/train_and_test_partitions/train_partition.h5",
    "/struct/mahamid/Irene/yeast/ribosomes/180426_005/G_sigma1/train_and_test_partitions/train_partition.h5",
    "/struct/mahamid/Irene/yeast/ribosomes/180426_021/G_sigma1/train_and_test_partitions/train_partition.h5",
    "/struct/mahamid/Irene/yeast/ribosomes/180426_024/G_sigma1/train_and_test_partitions/train_partition.h5",
    "/struct/mahamid/Irene/yeast/ribosomes/180711_003/G_sigma1/train_and_test_partitions/train_partition.h5",
    "/struct/mahamid/Irene/yeast/ribosomes/180711_004/G_sigma1/train_and_test_partitions/train_partition.h5",
    "/struct/mahamid/Irene/yeast/ribosomes/180711_005/G_sigma1/train_and_test_partitions/train_partition.h5",
    "/struct/mahamid/Irene/yeast/ribosomes/180711_018/G_sigma1/train_and_test_partitions/train_partition.h5",
    # "/struct/mahamid/Irene/yeast/ribosomes/180713_027/G_sigma1/train_and_test_partitions/data_aug_on_training_split.h5",
]

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-split", "--split",
                    help="path to tomogram to be segmented in hdf format",
                    type=float)
parser.add_argument("-label_name", "--label_name",
                    help="directory where the outputs will be stored",
                    type=str)
parser.add_argument("-model_nickname", "--model_nickname",
                    help="file where the outputs will be stored",
                    type=str)
parser.add_argument("-model_path", "--model_path",
                    help="name of category to be segmented",
                    type=str)
parser.add_argument("-log_dir", "--log_dir",
                    help="directory where the log files will be stored",
                    type=str)
parser.add_argument("-depth", "--depth",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-init_features", "--init_features",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-iterations", "--iterations",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-elu", "--elu",
                    help="name of category to be segmented",
                    type=bool)

args = parser.parse_args()
split = args.split
label_name = args.label_name
model_nickname = args.model_nickname
model_path = args.model_path
depth = args.depth
init_features = args.init_features
iterations = args.iterations
elu = args.elu
log_dir = args.log_dir

# Loading training and testing sets from different files
for n, training_data_path in enumerate(training_data_paths):
    print("Done loading training set from ", training_data_path)
    if n == 0:
        train_data, train_labels, val_data, val_labels, _ = \
            get_testing_and_training_sets_from_partition(training_data_path,
                                                         label_name,
                                                         split)
    else:
        train_data_tmp, train_labels_tmp, val_data_tmp, val_labels_tmp, _ = \
            get_testing_and_training_sets_from_partition(training_data_path,
                                                         label_name,
                                                         split)
        train_data = np.concatenate((train_data, train_data_tmp), axis=0)
        train_labels = np.concatenate((train_labels, train_labels_tmp), axis=0)
        val_data = np.concatenate((val_data, val_data_tmp), axis=0)
        val_labels = np.concatenate((val_labels, val_labels_tmp), axis=0)

######################### Unet preparation ##############################

# wrap into datasets
train_set = du.TensorDataset(torch.from_numpy(train_data),
                             torch.from_numpy(train_labels))
val_set = du.TensorDataset(torch.from_numpy(val_data),
                           torch.from_numpy(val_labels))

# wrap into data-loader
train_loader = du.DataLoader(train_set, shuffle=True,  # we shuffle before
                             batch_size=10)
val_loader = du.DataLoader(val_set, batch_size=10)

for test_index in range(iterations):

    net_confs = [
        {'depth': depth, 'initial_features': init_features, 'elu': elu},
    ]

    for conf in net_confs:
        print("Starting training for UNET with conf =", conf)
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
        model_name = str(test_index) + '_UNET_' + model_nickname + \
                     "_D" + str(conf['depth']) + \
                     "_IF" + str(conf['initial_features'])
        if elu:
            model_name += "_elu_"
        else:
            model_name += "_relu_"

    print("The model name is ", model_name)

    model_log_dir = join(log_dir, model_name)
    logger = TensorBoard(log_dir=model_log_dir, log_image_interval=1)
    print("The neural network training is now starting")
    n_epochs = 50
    model_name_pkl = model_name + ".pkl"
    model_path_pkl = join(model_path, model_name_pkl)
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
                                                 device=device,
                                                 step=step,
                                                 tb_logger=logger)

        if epoch == 0:
            validation_loss = current_validation_loss
        else:
            if current_validation_loss <= validation_loss:
                best_epoch = epoch
                validation_loss = current_validation_loss
                save_unet_model(path_to_model=model_path_pkl, epoch=epoch,
                                net=net, optimizer=optimizer, loss=loss)
            else:
                print("this model was not the best")

    model_name_txt = model_name + ".txt"
    data_txt_path = join(model_path, model_name_txt)
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
