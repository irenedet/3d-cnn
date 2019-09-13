import json
import pandas as pd
from distutils.util import strtobool

from src.python.filereaders.h5 import load_classification_training_set, \
    fill_multiclass_labels

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-dataset_table", "--dataset_table",
                    help="dataset_table",
                    type=str)
parser.add_argument("-semantic_classes", "--semantic_classes",
                    help="semantic_classes",
                    type=str)
parser.add_argument("-output_dir", "--output_dir",
                    help="output_dir",
                    type=str)
parser.add_argument("-box_side", "--box_side",
                    type=int)
parser.add_argument("-multi_label", "--multi_label",
                    type=str)
parser.add_argument("-tomo_name", "--tomo_name",
                    help="tomo_name in sessiondate/datanumber format",
                    type=str)
parser.add_argument("-co_labeling_dict", "--co_labeling_dict",
                    help="co_labeling_dict in .json file",
                    type=str, default=None)

args = parser.parse_args()
dataset_table = args.dataset_table
semantic_classes = args.semantic_classes
semantic_classes = semantic_classes.split(",")
box_side = args.box_side
output_dir = args.output_dir
tomo_name = args.tomo_name
multi_label = strtobool(args.multi_label)
co_labeling_dict_path = args.co_labeling_dict

df = pd.read_csv(dataset_table)
df['tomo_name'] = df['tomo_name'].astype(str)
path_to_output_h5 = df.loc[df['tomo_name'] == tomo_name, 'train_partition']

############## Load Dataset
subtomos_data, labels_data = load_classification_training_set(semantic_classes,
                                                              path_to_output_h5)
if multi_label:
    co_labeling_dict = json.loads(co_labeling_dict_path)
    # Check for dependencies (co-labeling):
    print("Multiple labels are allowed for a particle: multi-label case.")
    labels_data = fill_multiclass_labels(semantic_classes, co_labeling_dict,
                                         labels_data)
else:
    print("Only one class per particle: multi-class case.")
