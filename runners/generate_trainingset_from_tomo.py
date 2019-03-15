from os.path import join
from os import makedirs
# import argparse

from src.python.datasets.actions import partition_tomogram
from src.python.filereaders.hdf import _load_hdf_dataset
from src.python.osactions.filesystem import extract_file_name

# parser = argparse.ArgumentParser()
# parser.add_argument("-raw", "--path_to_raw",
#                     help="path to tomogram to be segmented in hdf format",
#                     type=str)
# parser.add_argument("-output", "--output_dir",
#                     help="directory where the outputs will be stored",
#                     type=str)
#
# args = parser.parse_args()
# path_to_raw = args.path_to_raw
# output_dir = args.output_dir

# 1. create hdf file with annotation from Sara:
# path_to_motl = "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/006/TM/motl_clean_4b.em"
# output_dir = "/scratch/trueba/3d-cnn/clean/"
# tomo_name = "180426_006"
path_to_motl = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_006/confs_16_5_bis_/motl_5182.csv"
output_dir = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_006/confs_16_5_bis_/"

# path_to_motl = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_005/confs_16_5_bis_/motl_4654.csv"
# output_dir = "/scratch/trueba/3d-cnn/cnn_evaluation/180426_005/confs_16_5_bis_/"

makedirs(name=output_dir, exist_ok=True)
hdf_output_path = join(output_dir, "motl_5182_ribos_mask.hdf")
z_shift = -330  # shift between original tomogram and subtomogram of analysis
output_shape = (321, 927, 927)

# from src.python.filewriters.h5 import write_hdf_particles_from_motl
#
# write_hdf_particles_from_motl(path_to_motl=path_to_motl,
#                               hdf_output_path=hdf_output_path,
#                               output_shape=output_shape,
#                               sphere_radius=8,
#                               values_in_motl=True,
#                               number_of_particles=None,
#                               z_shift=z_shift)

# 2. create subtomograms of raw and labeled
path_to_raw = "/scratch/trueba/3d-cnn/clean/180426_006/subtomo330-650.hdf"
# "/scratch/trueba/3d-cnn/clean/180426_005/subtomo370-620.hdf"
# "/scratch/trueba/3d-cnn/clean/181126_002/002_sq_df_sorted.hdf"
path_to_labeled = hdf_output_path
output_dir = "/scratch/trueba/3d-cnn/training_data/TEST/"
tomo_name = "180426_006"  # extract_file_name(path_to_file=path_to_raw)
output_dir = join(output_dir, tomo_name)
makedirs(name=output_dir, exist_ok=True)
output_h5_file_name = "ribosomes_training.h5"
output_h5_file_path = join(output_dir, output_h5_file_name)
subtomogram_shape = (128, 128, 128)
overlap = 12
label_name = "ribosomes"
#

# Fuctions that didn't work: ###########################################
# from src.python.filereaders.em import load_em_dataset
# header, raw_dataset = load_em_dataset(path_to_emfile=path_to_raw,
#                                       output_shape_xyz=(928, 928, 500))
# print("raw_dataset.shape", raw_dataset.shape)
########################################################################

raw_dataset = _load_hdf_dataset(hdf_file_path=path_to_raw)
labels_dataset = _load_hdf_dataset(hdf_file_path=path_to_labeled)
from src.python.datasets.actions import partition_raw_and_labels_tomograms

partition_raw_and_labels_tomograms(raw_dataset=raw_dataset,
                                   labels_dataset=labels_dataset,
                                   label_name=label_name,
                                   output_h5_file_path=output_h5_file_path,
                                   subtomo_shape=subtomogram_shape,
                                   overlap=overlap
                                   )
print("The script has finished!")
exit(output_h5_file_path)

