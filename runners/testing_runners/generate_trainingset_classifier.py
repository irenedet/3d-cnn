import argparse
from distutils.util import strtobool
from os.path import join

import pandas as pd

from file_actions.writers.h5 import generate_classification_training_set_per_tomo

parser = argparse.ArgumentParser()
parser.add_argument("-dataset_table", "--dataset_table",
                    help="dataset_table",
                    type=str)
parser.add_argument("-semantic_classes", "--semantic_classes",
                    help="semantic_classes",
                    type=str)
parser.add_argument("-write_on_table", "--write_on_table",
                    help="if True, name of training set will be recorded in db",
                    type=str)
parser.add_argument("-output_dir", "--output_dir",
                    help="output_dir",
                    type=str)
parser.add_argument("-box_side", "--box_side",
                    type=int)
parser.add_argument("-tomo_name", "--tomo_name",
                    help="tomo_name in sessiondate/datanumber format",
                    type=str)

args = parser.parse_args()
dataset_table = args.dataset_table
semantic_classes = args.semantic_classes
semantic_classes = semantic_classes.split(",")
box_side = args.box_side
write_on_table = strtobool(args.write_on_table)
output_dir = args.output_dir
tomo_name = args.tomo_name

path_to_output_h5 = join(output_dir, tomo_name)
path_to_output_h5 = join(path_to_output_h5, "training_set.h5")

generate_classification_training_set_per_tomo(dataset_table,
                                              tomo_name,
                                              semantic_classes,
                                              path_to_output_h5,
                                              box_side)

if write_on_table:
    print("path to training partition written on table: ", path_to_output_h5)
    df = pd.read_csv(dataset_table)
    df['tomo_name'] = df['tomo_name'].astype(str)
    df.loc[
        df['tomo_name'] == tomo_name, 'train_partition'] = path_to_output_h5
    df.to_csv(path_or_buf=dataset_table, index=False)
