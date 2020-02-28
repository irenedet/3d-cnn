#! /home/trueba/.conda/envs/mlcourse/bin/python3

import argparse
from os import makedirs
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as du
from src.python.networks.classes.unet import UNet
from src.python.networks.classes.visualizers import TensorBoard_multiclass

from networks.io import get_device
from networks.routines import train, validate
from src.python.tomogram_utils.actions import split_and_preprocess_dataset
from file_actions.readers import h5
from file_actions.writers.txt import write_model_description
from src.python.image.filters import preprocess_data

parser = argparse.ArgumentParser()
parser.add_argument("-data_path", "--training_data_path",
                    help="path to training set in h5 format",
                    type=str)
parser.add_argument("-label", "--label_name",
                    help="label name of segmentation class",
                    type=str)
parser.add_argument("-log_dir", "--log_dir",
                    help="logs directory where training losses will be stored",
                    type=str)
parser.add_argument("-model_name", "--model_initial_name",
                    help="model distinctive name",
                    type=str)
parser.add_argument("-model_path", "--model_path",
                    help="directory where the model will be stored",
                    type=str)
parser.add_argument("-split", "--split",
                    help="split between training and validation sets",
                    type=int)
parser.add_argument("-classes", "--output_classes",
                    help="number of output classes to be segmented",
                    type=int)
parser.add_argument("-n_epochs", "--number_of_epochs",
                    help="number of epoches for training",
                    type=int)
parser.add_argument("-weights", "--weights_per_class",
  type=str,
  )


args = parser.parse_args()
training_data_path = args.training_data_path
label_name = args.label_name
split = args.split
log_dir = args.log_dir
model_initial_name = args.model_initial_name
model_path = args.model_path
output_classes = args.output_classes
n_epochs = args.number_of_epochs
weight = args.weights_per_class
weight = list(map(float, weight.split(',')))

makedirs(name=log_dir, exist_ok=True)
makedirs(name=model_path, exist_ok=True)

net_confs = [
    {'final_activation': nn.LogSoftmax(dim=1),
     'depth': 3,
     'initial_features': 8,
     "out_channels": output_classes},
    {'final_activation': nn.LogSoftmax(dim=1),
     'depth': 4,
     'initial_features': 8,
     "out_channels": output_classes}
]

print("*************************************")
print("The cnn_training.py script is running")
print("*************************************")

# check if we have  a gpu
device = get_device()
print("The training data path is ", training_data_path)

raw_data, labels = h5.read_training_data(training_data_path,
                                         label_name=label_name)

print("Initial unique labels", np.unique(labels))

# Normalize data
preprocessed_data = preprocess_data(raw_data)

labels = np.array(labels, dtype=np.long)

train_data, train_labels, val_data, val_labels, data_order = \
    split_and_preprocess_dataset(preprocessed_data, labels, split)

train_data = train_data[:, None]
val_data = val_data[:, None]

print("train_data.shape", train_data.shape)
print("train_labels.shape", train_labels.shape)

# wrap into datasets
train_set = du.TensorDataset(torch.from_numpy(train_data),
                             torch.from_numpy(train_labels))
val_set = du.TensorDataset(torch.from_numpy(val_data),
                           torch.from_numpy(val_labels))

# wrap into data-loader (we shuffle before during training test construction)
train_loader = du.DataLoader(train_set, shuffle=True, batch_size=5)
val_loader = du.DataLoader(val_set, batch_size=5)

for test_index in range(1):
    for conf in net_confs:
        net = UNet(**conf)
        net = net.to(device)
        weight_tensor = torch.tensor(weight).to(device)
        loss = nn.NLLLoss(weight=weight_tensor)
        loss = loss.to(device)
        optimizer = optim.Adam(net.parameters())

        # ToDo build the dice coefficient metric
        metric = loss

        # built tensorboard logger
        model_name = model_initial_name + label_name + "_D_" + \
                     str(conf['depth']) + "_IF_" + \
                     str(conf['initial_features'])
        model_name_pkl = model_name + ".pkl"
        model_path_pkl = join(model_path, model_name_pkl)
        model_name_txt = model_name + ".txt"
        data_txt_path = join(model_path, model_name_txt)
        write_model_description(data_txt_path, training_data_path, label_name,
                                split, model_name_pkl, conf, data_order)

        log_model = join(log_dir, model_name)
        logger = TensorBoard_multiclass(log_dir=log_model, log_image_interval=1)
        print("The neural network training is now starting")
        for epoch in range(n_epochs):
            # apply training for one epoch
            train(net, train_loader, optimizer=optimizer, loss_function=loss,
                  epoch=epoch, device=device, log_interval=1, tb_logger=logger)
            step = epoch * len(train_loader.dataset)
            # run validation after training epoch
            current_validation_loss = validate(net, val_loader, loss, metric, device=device, step=step,
                     tb_logger=logger)




            if epoch == 0:
                validation_loss = current_validation_loss
            else:
                if current_validation_loss <= validation_loss:
                    torch.save(net.state_dict(), model_path_pkl)
                    validation_loss = current_validation_loss
                else:
                    print("this model was not the best")

print("We have finished the training!")
