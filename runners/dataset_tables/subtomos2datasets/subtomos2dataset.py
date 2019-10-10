from os.path import join
import pandas as pd

from src.python.filewriters.h5 import \
    write_dataset_from_subtomos_with_overlap_multiclass, \
    write_clustering_labels_subtomos
from src.python.naming import h5_internal_paths
from src.python.utils.cast_types import string_to_list
from distutils.util import strtobool

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-box_length", "--box_length",
                    help="subtomogram side length (cubic subtomograms assumed)",
                    type=int)
parser.add_argument("-label_name", "--label_name",
                    help="directory where the outputs will be stored",
                    type=str)
parser.add_argument("-overlap", "--overlap",
                    help="Thickness of subtomo overlap in pixels",
                    type=int)
parser.add_argument("-class_number", "--class_number",
                    help="class number to re-construct",
                    type=int)
parser.add_argument("-output_path", "--output_path",
                    help="directory where the output hdf file will be stored",
                    type=str)
parser.add_argument("-cluster_labels", "--cluster_labels",
                    help="directory where the output hdf file will be stored",
                    type=str, default=False)
parser.add_argument("-reconstruction_type", "--reconstruction_type",
                    help="either prediction or labels to define reconst. set",
                    type=str, default="prediction")
# NEW
parser.add_argument("-dataset_table", "--dataset_table",
                    help="path to dataset_table",
                    type=str)
parser.add_argument("-tomo_name", "--tomo_name",
                    help="tomo to be analyzed",
                    type=str)

args = parser.parse_args()
label_name = args.label_name
box_length = args.box_length
overlap_thickness = args.overlap
class_number = args.class_number
output_path = args.output_path
cluster_labels = strtobool(args.cluster_labels)
subtomo_shape = (box_length, box_length, box_length)

reconstruction_type = args.reconstruction_type

# NEW
dataset_table = args.dataset_table
tomo_name = args.tomo_name
df = pd.read_csv(dataset_table)

df['tomo_name'] = df['tomo_name'].astype(str)
tomo_df = df[df['tomo_name'] == tomo_name]
x_dim = int(tomo_df.iloc[0]['x_dim'])
y_dim = int(tomo_df.iloc[0]['y_dim'])
z_dim = int(tomo_df.iloc[0]['z_dim'])
test_partition = tomo_df.iloc[0]['test_partition']

output_shape = (z_dim, y_dim, x_dim)

print(reconstruction_type)
assert reconstruction_type in ["prediction", "labels"]

if reconstruction_type == "prediction":
    subtomos_internal_path = join(
        h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS,
        label_name)
else:
    subtomos_internal_path = join(
        h5_internal_paths.LABELED_SUBTOMOGRAMS,
        label_name)

print(subtomos_internal_path)
if not cluster_labels:
    write_dataset_from_subtomos_with_overlap_multiclass(
        output_path,
        test_partition,
        output_shape,
        subtomo_shape,
        subtomos_internal_path,
        class_number,
        overlap_thickness,
        reconstruction_type=reconstruction_type)
else:
    subtomos_internal_path = h5_internal_paths.CLUSTERING_LABELS
    write_clustering_labels_subtomos(
        output_path,
        test_partition,
        output_shape,
        subtomo_shape,
        subtomos_internal_path,
        label_name,
        class_number,
        overlap_thickness)
