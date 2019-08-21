import numpy as np
import pandas as pd
from os import makedirs
from os.path import join
from src.python.filereaders.datasets import load_dataset
from src.python.filereaders.csv import read_motl_from_csv
from src.python.filewriters.csv import motl_writer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-output_dir", "--output_dir",
                    help="path to the output directory",
                    type=str)
parser.add_argument("-tomo_name", "--tomo_name",
                    help="name of tomogram in format sessionname/datasetnumber",
                    type=str)
parser.add_argument("-dataset_table", "--dataset_table",
                    help="path to dataset table",
                    type=str)
parser.add_argument("-csv_motl", "--csv_motl",
                    help="path to motl in csv format",
                    type=str)
parser.add_argument("-border_xy", "--dataset_border_xy",
                    help="border thickness in xy plane to discard particles",
                    type=int)
parser.add_argument("-lamella_extension", "--lamella_extension",
                    help="pixels up and down the lamella that will be added",
                    type=int)

args = parser.parse_args()
csv_motl = args.csv_motl
dataset_table = args.dataset_table
tomo_name = args.tomo_name
dataset_border_xy = args.dataset_border_xy
lamella_extension = args.lamella_extension
output_dir = args.output_dir

df = pd.read_csv(dataset_table)
df['tomo_name'] = df['tomo_name'].astype(str)
tomo_df = df[df['tomo_name'] == tomo_name]
z_shift = int(tomo_df.iloc[0]['z_shift'])
x_dim = int(tomo_df.iloc[0]['x_dim'])
y_dim = int(tomo_df.iloc[0]['y_dim'])
z_dim = int(tomo_df.iloc[0]['z_dim'])
lamella_file = tomo_df.iloc[0]['lamella_file']
print("lamella_file =", lamella_file)
conserved_points_dir = join(output_dir, "in_lamella")
discarded_points_dir = join(output_dir, "outside_lamella")
makedirs(name=conserved_points_dir, exist_ok=True)
makedirs(name=discarded_points_dir, exist_ok=True)

lamella_indicator = load_dataset(path_to_dataset=lamella_file)
motl_predicted = read_motl_from_csv(path_to_csv_motl=csv_motl)

lamella_indicator = np.array(lamella_indicator)

motl_values = [row[0] for row in motl_predicted]
predicted_coordinates = [np.array([row[7], row[8], row[9]]) for row in
                         motl_predicted]

conserved_points = []
conserved_values = []
discarded_points = []
discarded_values = []
for value, point in zip(motl_values, predicted_coordinates):
    point = [int(entry) for entry in point]
    x, y, z = point
    z_up = z - z_shift + lamella_extension
    z_down = z - z_shift - lamella_extension
    lamella_border_up = np.min([z_up, z_dim - 1])
    lamella_border_down = np.max([z_down, 0])
    if lamella_indicator[z - z_shift, y, x] == 1 and np.min(
            [x, y, x_dim - x, y_dim - y]) > dataset_border_xy:
        conserved_values += [value]
        conserved_points += [point]
    elif lamella_indicator[lamella_border_up, y, x] == 1 and np.min(
            [x, y, x_dim - x, y_dim - y]) > dataset_border_xy:
        conserved_values += [value]
        conserved_points += [point]
    elif lamella_indicator[lamella_border_down, y, x] == 1 and np.min(
            [x, y, x_dim - x, y_dim - y]) > dataset_border_xy:
        conserved_values += [value]
        conserved_points += [point]
    else:
        discarded_points += [point]
        discarded_values += [value]

motl_writer(path_to_output_folder=conserved_points_dir,
            list_of_peak_scores=conserved_values,
            list_of_peak_coords=conserved_points,
            in_tom_format=True)
motl_writer(path_to_output_folder=discarded_points_dir,
            list_of_peak_scores=discarded_values,
            list_of_peak_coords=discarded_points,
            in_tom_format=True)
