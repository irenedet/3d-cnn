import argparse
import os
from distutils.util import strtobool
from os import makedirs

import pandas as pd

from constants.dataset_tables import DatasetTableHeader
from file_actions.writers.h5 import write_particle_mask_from_motl

parser = argparse.ArgumentParser()
parser.add_argument("-dataset_table", "--dataset_table",
                    help="path to dataset_table in .csv format",
                    type=str)
parser.add_argument("-tomo_name", "--tomo_name",
                    help="tomo_name in sessiondate/datanumber format",
                    type=str)
parser.add_argument("-class_name", "--class_name",
                    help="class name either ribo or fas",
                    type=str)
parser.add_argument("-output_path", "--output_path",
                    help="file path of output mask",
                    type=str)
parser.add_argument("-radius", "--sphere_radius",
                    type=int)
parser.add_argument("-coords_in_tom_format", "--coords_in_tom_format",
                    type=str)
parser.add_argument("-write_on_dataset_table", "--write_on_dataset_table",
                    type=str)
parser.add_argument("-values_in_motl", "--values_in_motl",
                    help="If True, score value of motl is assigned to mask" +
                         "otherwise, 1 is assigned.",
                    default="True",
                    type=str)

args = parser.parse_args()
tomo_name = args.tomo_name
dataset_table = args.dataset_table
class_name = args.class_name
coords_in_tom_format = strtobool(args.coords_in_tom_format)
radius = args.sphere_radius
output_path = args.output_path
values_in_motl = strtobool(args.values_in_motl)
write_on_dataset_table = strtobool(args.write_on_dataset_table)
semantic_classes = [class_name]

DTHeader = DatasetTableHeader(semantic_classes=semantic_classes)
df = pd.read_csv(dataset_table)
df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)
tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
z_shift = int(tomo_df.iloc[0][DTHeader.z_shift])
x_dim = int(tomo_df.iloc[0][DTHeader.x_dim])
y_dim = int(tomo_df.iloc[0][DTHeader.y_dim])
z_dim = int(tomo_df.iloc[0][DTHeader.z_dim])
clean_motive_list_name = DTHeader.clean_motls[0]

path_to_motl = tomo_df.iloc[0][clean_motive_list_name]

print("coords_in_tom_format = ", coords_in_tom_format)
output_dir = os.path.dirname(output_path)
makedirs(output_dir, exist_ok=True)

output_shape = (z_dim, y_dim, x_dim)

write_particle_mask_from_motl(path_to_motl=path_to_motl,
                              output_path=output_path,
                              output_shape=output_shape,
                              sphere_radius=radius,
                              values_in_motl=values_in_motl,
                              number_of_particles=None,
                              z_shift=z_shift,
                              particles_in_tom_format=coords_in_tom_format)

if write_on_dataset_table:
    clean_mask_name = DTHeader.masks_names[0]
    df.loc[df[DTHeader.tomo_name] == tomo_name, clean_mask_name] = [output_path]
    df.to_csv(path_or_buf=dataset_table, index=False)
