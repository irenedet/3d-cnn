#! /home/trueba/.conda/envs/mlcourse/bin/python3

import argparse
from distutils.util import strtobool
from os import makedirs
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as du
from src.python.networks.routines import train, validate
from src.python.networks.unet import UNet_dropout
from src.python.networks.visualizers import TensorBoard_multiclass

from src.python.filewriters.csv import write_on_models_notebook
from src.python.networks.io import get_device
from src.python.networks.loss import DiceCoefficientLoss
from src.python.datasets.actions import split_dataset
from src.python.filereaders import h5
from src.python.filewriters.txt import write_model_description
from src.python.image.filters import preprocess_data
from src.python.networks.utils import save_unet_model, load_unet_model

parser = argparse.ArgumentParser()
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
                    type=float)
parser.add_argument("-classes", "--output_classes",
                    help="number of output classes to be segmented",
                    type=int)
parser.add_argument("-n_epochs", "--number_of_epochs",
                    help="number of epoches for training",
                    type=int)
parser.add_argument("-weights", "--weights_per_class",
                    type=str,
                    default='1,1,1')
parser.add_argument("-retrain", "--retrain",
                    default="False",
                    type=str)
parser.add_argument("-path_to_old_model", "--path_to_old_model",
                    default=False,
                    type=str)
parser.add_argument("-segmentation_names", "--segmentation_names",
                    default='ribo,fas,memb',
                    type=str)
parser.add_argument("-depth", "--depth",
                    type=int)
parser.add_argument("-initial_features", "--initial_features",
                    type=int)
parser.add_argument("-dropout", "--dropout",
                    type=int)
parser.add_argument("-training_paths_list", "--training_paths_list",
                    type=str)
parser.add_argument("-models_notebook", "--models_notebook",
                    default="None",
                    type=str)
parser.add_argument("-skip", "--skip",
                    type=int)

args = parser.parse_args()
training_paths_list = args.training_paths_list
training_paths_list = training_paths_list.split('\n')
skip_training = args.skip
label_name = args.label_name
split = args.split
log_dir = args.log_dir
model_initial_name = args.model_initial_name
model_path = args.model_path
output_classes = args.output_classes
n_epochs = args.number_of_epochs
models_notebook_path = args.models_notebook
retrain = strtobool(args.retrain)
path_to_old_model = args.path_to_old_model
depth = args.depth
initial_features = args.initial_features
weight = args.weights_per_class
dropout = args.dropout
weight = list(map(float, weight.split(',')))
segmentation_names = args.segmentation_names
segmentation_names = list(map(str, segmentation_names.split(',')))

makedirs(name=log_dir, exist_ok=True)
makedirs(name=model_path, exist_ok=True)

final_activation = nn.Sigmoid()
net_confs = [
    {'final_activation': final_activation,
     'depth': depth,
     'initial_features': initial_features,
     "out_channels": output_classes,
     "dropout": dropout},
]

print("*************************************")
print("The cnn_training.py script is running")
print("*************************************")

print(training_paths_list)

# check if we have  a gpu
device = get_device()

for n, training_data_path in enumerate(training_paths_list):
    print("\n")
    if skip_training == n:
        print("Skipping ", training_data_path)
    else:
        print("Loading training set from ", training_data_path)
        if 'train_data' not in locals():
            raw_data, labels = h5.read_training_data_dice_multi_class(
                training_data_path=training_data_path,
                segmentation_names=segmentation_names,
                split=-1)
            print(training_data_path)
            print("labels.shape = ", labels.shape)
            print("raw_data.shape = ", raw_data.shape)
            print("Initial unique labels", np.unique(labels))

            # Normalize data
            preprocessed_data = preprocess_data(raw_data)
            preprocessed_data = np.array(preprocessed_data)[:, None]
            labels = np.array(labels, dtype=np.long)

            train_data, train_labels, val_data, val_labels, _ = \
                split_dataset(preprocessed_data, labels, split)

            print("train_data.shape", train_data.shape)
            print("train_labels.shape", train_labels.shape)

        else:
            raw_data, labels = h5.read_training_data_dice_multi_class(
                training_data_path=training_data_path,
                segmentation_names=segmentation_names,
                split=-1)
            print(training_data_path)
            print("labels.shape = ", labels.shape)
            print("raw_data.shape = ", raw_data.shape)
            print("Initial unique labels", np.unique(labels))

            # Normalize data
            preprocessed_data = preprocess_data(raw_data)
            preprocessed_data = np.array(preprocessed_data)[:, None]
            labels = np.array(labels, dtype=np.long)

            train_data_tmp, train_labels_tmp, val_data_tmp, val_labels_tmp, _ = \
                split_dataset(preprocessed_data, labels, split)

            print("train_data.shape", train_data.shape)
            print("train_labels.shape", train_labels.shape)

            train_data = np.concatenate((train_data, train_data_tmp), axis=0)
            train_labels = np.concatenate((train_labels, train_labels_tmp),
                                          axis=0)
            val_data = np.concatenate((val_data, val_data_tmp), axis=0)
            val_labels = np.concatenate((val_labels, val_labels_tmp), axis=0)

train_set = du.TensorDataset(torch.from_numpy(train_data),
                             torch.from_numpy(train_labels))
val_set = du.TensorDataset(torch.from_numpy(val_data),
                           torch.from_numpy(val_labels))

# wrap into data-loader (shuffle in False works better apparently)
train_loader = du.DataLoader(train_set, shuffle=False, batch_size=5)
val_loader = du.DataLoader(val_set, batch_size=5)

for conf in net_confs:
    if retrain:
        net, optimizer, old_epoch, loss = load_unet_model(
            path_to_model=path_to_old_model,
            confs=conf,
            mode="train")
        net = net.to(device)
        loss = loss.to(device)
    else:
        net = UNet_dropout(**conf)
        net = net.to(device)
        loss = DiceCoefficientLoss()
        loss = loss.to(device)
        optimizer = optim.Adam(net.parameters())
        old_epoch = 0

    metric = loss
    model_name = model_initial_name + label_name + "_D_" + \
                 str(conf['depth']) + "_IF_" + \
                 str(conf['initial_features']) + \
                 "drop" + str(conf['dropout'])
    model_name_pkl = model_name + ".pkl"
    model_path_pkl = join(model_path, model_name_pkl)
    model_name_txt = model_name + ".txt"
    data_txt_path = join(model_path, model_name_txt)
    write_model_description(file_path=data_txt_path,
                            training_data_path=str(training_paths_list),
                            label_name=label_name, split=split,
                            model_name_pkl=model_path_pkl, conf=conf,
                            skip_training_set=skip_training)
    log_model = join(log_dir, model_name)
    logger = TensorBoard_multiclass(log_dir=log_model, log_image_interval=1)
    print("The neural network training is now starting")

    write_on_models_notebook(model_name, model_path_pkl, log_model, depth,
                             initial_features, n_epochs, training_paths_list,
                             split, output_classes, segmentation_names, retrain,
                             path_to_old_model, models_notebook_path, dropout)

    for epoch in range(n_epochs):
        # apply training for one epoch
        new_epoch = epoch + old_epoch
        train(net, train_loader, optimizer=optimizer, loss_function=loss,
              epoch=new_epoch, device=device, log_interval=1, tb_logger=logger,
              log_image=False)
        step = new_epoch * len(train_loader.dataset)
        # run validation after training epoch
        current_validation_loss = validate(net, val_loader, loss, metric,
                                           device=device, step=step,
                                           tb_logger=logger)
        if epoch == 0:
            validation_loss = current_validation_loss
            best_epoch = 0
        else:
            if current_validation_loss <= validation_loss:
                best_epoch = new_epoch
                validation_loss = current_validation_loss
                save_unet_model(path_to_model=model_path_pkl, epoch=new_epoch,
                                net=net, optimizer=optimizer, loss=loss)
            else:
                print("this model was not the best")

print("We have finished the training!")
print("The best validation loss was", validation_loss)
