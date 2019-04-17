from os.path import join
from os import makedirs
import argparse
#
parser = argparse.ArgumentParser()
parser.add_argument("-motl", "--path_to_motl",
                    help="path to motive list in .em or .csv format",
                    type=str)
parser.add_argument("-output", "--output_dir",
                    help="directory where the outputs will be stored",
                    type=str)
parser.add_argument("-radius", "--sphere_radius",
                    type=int)
parser.add_argument("-z_shift", "--output_z_shift",
                    type=int)
parser.add_argument("-shape_x", "--output_shape_x",
                    type=int)
parser.add_argument("-shape_y", "--output_shape_y",
                    type=int)
parser.add_argument("-shape_z", "--output_shape_z",
                    type=int)

args = parser.parse_args()
path_to_motl = args.path_to_motl
output_dir = args.output_dir
shape_x = args.output_shape_x
shape_y = args.output_shape_y
shape_z = args.output_shape_z
z_shift = args.output_z_shift
radius = args.sphere_radius

makedirs(name=output_dir, exist_ok=True)
hdf_output_path = join(output_dir, "particles_mask.hdf")
# z_shift = -330  # shift between original tomogram and subtomogram of analysis
output_shape = (shape_z, shape_y, shape_x)

from src.python.filewriters.h5 import write_hdf_particles_from_motl

write_hdf_particles_from_motl(path_to_motl=path_to_motl,
                              hdf_output_path=hdf_output_path,
                              output_shape=output_shape,
                              sphere_radius=radius,
                              values_in_motl=True,
                              number_of_particles=None,
                              z_shift=z_shift,
                              particles_in_tom_format=False)


