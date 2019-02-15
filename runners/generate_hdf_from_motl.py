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

args = parser.parse_args()
path_to_motl = args.path_to_motl
output_dir = args.output_dir

# path_to_motl = "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/006/TM/motl_clean_4b.em"


path_to_motl = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_006/confs_16_5_bis_/motl_5182.csv"
output_dir = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_006/confs_16_5_bis_/"

# path_to_motl = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_005/confs_16_5_bis_/motl_4654.csv"
# output_dir = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_005/confs_16_5_bis_/"

makedirs(name=output_dir, exist_ok=True)
hdf_output_path = join(output_dir, "motl_5182_ribos_mask.hdf")
z_shift = -330  # shift between original tomogram and subtomogram of analysis
output_shape = (321, 927, 927)

from src.python.filewriters.h5 import write_hdf_particles_from_motl

write_hdf_particles_from_motl(path_to_motl=path_to_motl,
                              hdf_output_path=hdf_output_path,
                              output_shape=output_shape,
                              sphere_radius=8,
                              values_in_motl=True,
                              number_of_particles=None,
                              z_shift=z_shift)
