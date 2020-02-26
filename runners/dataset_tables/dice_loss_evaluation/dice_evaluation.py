import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from filereaders.datasets import load_dataset
from filewriters.csv import write_statistics
from filewriters.h5 import write_dataset_hdf
from pytorch_cnn.classes.loss import DiceCoefficient

parser = argparse.ArgumentParser()
parser.add_argument("-tomo_name", "--tomo_name",
                    help="name of tomogram in format sessionname/datasetnumber",
                    type=str)
parser.add_argument("-dataset_table", "--dataset_table",
                    help="path to dataset table",
                    type=str)
parser.add_argument("-statistics_file", "--statistics_file",
                    help="file where auPRC will be recorded.",
                    type=str)
parser.add_argument("-class_number", "--class_number",
                    help="class number",
                    type=int)
parser.add_argument("-semantic_classes", "--semantic_classes",
                    help="semantic_classes separated by commas",
                    type=str)
parser.add_argument("-label_name", "--label_name",
                    help="name of class, i.e. ribo or fas",
                    type=str)
parser.add_argument("-prediction_path", "--prediction_path",
                    help="prediction dataset path",
                    type=str)
parser.add_argument("-target_path", "--target_path",
                    help="the dataset to compare with",
                    type=str)

args = parser.parse_args()
dataset_table = args.dataset_table
tomo_name = args.tomo_name
# output_dir = args.output_dir

label_name = args.label_name
class_number = args.class_number
statistics_file = args.statistics_file
prediction_path = args.prediction_path
target_path = args.target_path

semantic_classes = args.semantic_classes
semantic_classes = semantic_classes.split(',')

class_name = semantic_classes[class_number]
clean_mask_name = class_name + '_mask'

df = pd.read_csv(dataset_table)
df['tomo_name'] = df['tomo_name'].astype(str)
tomo_df = df[df['tomo_name'] == tomo_name]
x_dim = int(tomo_df.iloc[0]['x_dim'])
y_dim = int(tomo_df.iloc[0]['y_dim'])
z_dim = int(tomo_df.iloc[0]['z_dim'])

lamella_file = tomo_df.iloc[0]['lamella_file']
if target_path == "None":
    target_path = tomo_df.iloc[0][clean_mask_name]

if str(lamella_file) == "nan":
    prediction = load_dataset(prediction_path)
else:
    lamella_indicator = load_dataset(path_to_dataset=lamella_file)
    prediction = load_dataset(path_to_dataset=prediction_path)
    shx, shy, shz = [np.min([shl, shp]) for shl, shp in
                     zip(lamella_indicator.shape, prediction.shape)]
    lamella_indicator = lamella_indicator[:shx, :shy, :shz]
    prediction = prediction[:shx, :shy, :shz]
    prediction = np.array(lamella_indicator, dtype=float) * np.array(prediction,
                                                                     dtype=float)

target = load_dataset(path_to_dataset=target_path)

sigmoid = nn.Sigmoid()
shx, shy, shz = [np.min([shl, shp]) for shl, shp in
                 zip(target.shape, prediction.shape)]

target = target[:shx, :shy, :shz]
prediction = prediction[:shx, :shy, :shz]

prediction = sigmoid(torch.from_numpy(prediction).float())
prediction = 1 * (prediction == 1).float()
target = torch.from_numpy(target).float()

measure = DiceCoefficient()
dice_statistic = measure.forward(prediction=prediction, target=target)
prediction = prediction.numpy()
sigmoid_prediction_path = os.path.dirname(prediction_path)
sigmoid_prediction_path = os.path.join(sigmoid_prediction_path,
                                       "sigmoid_prediction.hdf")
write_dataset_hdf(output_path=sigmoid_prediction_path, tomo_data=prediction)
dice_statistic = float(dice_statistic)
statistics_label = label_name + "_dice"
write_statistics(statistics_file=statistics_file,
                 statistics_label=statistics_label,
                 tomo_name=tomo_name,
                 stat_measure=dice_statistic)

print("Dice coefficient =", dice_statistic)
