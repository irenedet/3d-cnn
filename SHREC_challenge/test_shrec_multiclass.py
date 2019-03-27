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
from pytorch_cnn.classes.unet_new import UNet

from pytorch_cnn.classes.visualizers import TensorBoard, TensorBoard_multiclass
from pytorch_cnn.classes.routines import train, validate
from pytorch_cnn.io import get_device

print("*************************************")
print("The cnn_training.py script is running")
print("*************************************")

# check if we have  a gpu
device = get_device()

training_data_path = \
    "/scratch/trueba/shrec/0_sph_masks/training_sets/top5_foreground_training.h5"
# "/scratch/trueba/shrec/0_test/ALL_particles_big_spheres_training.h5"
# "/scratch/trueba/shrec/0_test/ALL_particles_training.h5"
# "/scratch/trueba/shrec/0_test/particle1_2_training.h5"
# '/scratch/trueba/3d-cnn/training_data/ribosomes/ribo_training_grid.h5'
# '/scratch/trueba/3d-cnn/training_data/training_data_side128_49examples.h5'
# '/scratch/trueba/3d-cnn/training_data/ribosomes/ribo_training_grid.h5'
# '/scratch/trueba/3d-cnn/training_data/ribosomes/ribo_training_grid.h5'
# label_name = "ALL_particles"
label_name = "all_particles"
split = 850
print("The training data path is ", training_data_path)

raw_data, labels = h5.read_training_data(training_data_path,
                                         label_name=label_name)

print("Initial unique labels", np.unique(labels))

view_data = False

if view_data:
    view_images_h5(data_path=training_data_path, img_range=(5, 10))
else:
    print(
        "The data viewer is deactivated, to activate it set view_data to True")

# Normalize data
preprocessed_data = preprocess_data(raw_data)

# add channel dimension only to input data
# preprocessed_data = np.array(preprocessed_data)[:, None]
labels = np.array(labels, dtype=np.long)

train_data, train_labels, val_data, val_labels, data_order = \
    split_dataset(preprocessed_data, labels, split)

train_data = train_data[:, None]
val_data = val_data[:, None]

print("train_data.shape", train_data.shape)
print("train_labels.shape", train_labels.shape)

# wrap into datasets
train_set = du.TensorDataset(torch.from_numpy(train_data),
                             torch.from_numpy(train_labels))
val_set = du.TensorDataset(torch.from_numpy(val_data),
                           torch.from_numpy(val_labels))

# wrap into data-loader
train_loader = du.DataLoader(train_set, shuffle=False,  # we shuffle before
                             batch_size=5)
val_loader = du.DataLoader(val_set, batch_size=5)
classes = 2
for test_index in range(1):
    net_confs = [
        {'final_activation': nn.LogSoftmax(dim=1),
         'depth': 2,
         'initial_features': 8,
         "out_channels": classes},
        {'final_activation': nn.LogSoftmax(dim=1),
         'depth': 3,
         'initial_features': 8,
         "out_channels": classes},
        {'final_activation': nn.LogSoftmax(dim=1),
         'depth': 2,
         'initial_features': 16,
         "out_channels": classes},
        {'final_activation': nn.LogSoftmax(dim=1),
         'depth': 3,
         'initial_features': 16,
         "out_channels": classes},
    ]

    for conf in net_confs:
        net = UNet(**conf)
        net = net.to(device)
        weight = [1 for n in range(classes)]
        weight[0] = 0.01  # background
        weight_tensor = torch.tensor(weight).to(device)
        loss = nn.NLLLoss(weight=weight_tensor)
        loss = loss.to(device)
        optimizer = optim.Adam(net.parameters())

        # build the dice coefficient metric
        metric = loss
        # metric = metric.to(device)

        # built tensorboard logger
        model_name = "Unet_top5_fore_sph_adjusted_radius_" + label_name + "_D_" + \
                     str(conf['depth']) + "_IF_" + \
                     str(conf['initial_features'])
        log_dir = join('shrec_sph_logs/', model_name)
        logger = TensorBoard_multiclass(log_dir=log_dir, log_image_interval=1)
        print("The neural network training is now starting")
        n_epochs = 200
        for epoch in range(n_epochs):
            # apply training for one epoch
            train(net, train_loader, optimizer=optimizer, loss_function=loss,
                  epoch=epoch, device=device, log_interval=1, tb_logger=logger)
            step = epoch * len(train_loader.dataset)
            # run validation after training epoch
            validate(net, val_loader, loss, metric, device=device, step=step,
                     tb_logger=logger)
        model_name_pkl = model_name + ".pkl"
        model_path = join("./shrec_models/multi-class", model_name_pkl)
        torch.save(net.state_dict(), model_path)

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
