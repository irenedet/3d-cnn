from os.path import join
from os import makedirs
import argparse
from file_actions.writers.h5 import write_particle_mask_from_motl_in_score_range

# from file_actions.writers.h5 import write_particle_mask_from_motl

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

# path_to_motl = "/struct/mahamid/Irene/yeast/healthy/180426/004/fas/motl/corrected_motl_191108_shifted_by_16.csv"
# output_dir = "/struct/mahamid/Irene/yeast/healthy/180426/004/fas/motl/"
# path_to_motl = "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/006/TM/motl_clean_4b.em"

path_to_motl = "/struct/mahamid/Irene/yeast/quantifications/180426/021/ribo_motl_with_nn_dist_score.csv"
output_dir = "/struct/mahamid/Irene/yeast/quantifications/180426/021"
makedirs(name=output_dir, exist_ok=True)
hdf_output_path = join(output_dir, "only_in_polysome_peak.hdf")
z_shift = 0  # shift between original tomogram and subtomogram of analysis
output_shape = (1000, 928, 960)

write_particle_mask_from_motl_in_score_range(path_to_motl=path_to_motl,
                                             output_path=hdf_output_path,
                                             output_shape=output_shape,
                                             sphere_radius=8,
                                             score_range=(1, 14),
                                             number_of_particles=None,
                                             z_shift=z_shift,
                                             particles_in_tom_format=True)
