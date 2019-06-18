import os
from os import makedirs
from src.python.filewriters.h5 import write_hdf_particles_from_motl
import argparse

#
parser = argparse.ArgumentParser()
parser.add_argument("-motl", "--path_to_motl",
                    help="path to motive list in .em or .csv format",
                    type=str)
parser.add_argument("-hdf_output_path", "--hdf_output_path",
                    help="directory where the outputs will be stored",
                    type=str)
parser.add_argument("-radius", "--sphere_radius",
                    type=int)
parser.add_argument("-z_shift", "--output_z_shift",
                    type=int)
parser.add_argument("-coords_in_tom_format", "--coords_in_tom_format",
                    type=bool)
parser.add_argument("-shape_x", "--output_shape_x",
                    type=int)
parser.add_argument("-shape_y", "--output_shape_y",
                    type=int)
parser.add_argument("-shape_z", "--output_shape_z",
                    type=int)

args = parser.parse_args()
path_to_motl = args.path_to_motl
shape_x = args.output_shape_x
shape_y = args.output_shape_y
shape_z = args.output_shape_z
z_shift = args.output_z_shift
coords_in_tom_format = args.coords_in_tom_format
radius = args.sphere_radius
hdf_output_path = args.hdf_output_path

print("coords_in_tom_format = ", coords_in_tom_format)
output_dir = os.path.dirname(hdf_output_path)
makedirs(name=output_dir, exist_ok=True)

# z_shift = -330  # shift between original tomogram and subtomogram of analysis
output_shape = (shape_z, shape_y, shape_x)

write_hdf_particles_from_motl(path_to_motl=path_to_motl,
                              hdf_output_path=hdf_output_path,
                              output_shape=output_shape,
                              sphere_radius=radius,
                              values_in_motl=True,
                              number_of_particles=None,
                              z_shift=z_shift,
                              particles_in_tom_format=True)
