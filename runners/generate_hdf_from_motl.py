import argparse
import os.path

import numpy as np

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

# args = parser.parse_args()
# path_to_motl = args.path_to_motl
# output_dir = args.output_dir

# path_to_motl = "/struct/mahamid/Irene/yeast/healthy/180426/004/fas/motl/corrected_motl_191108_shifted_by_16.csv"
# output_dir = "/struct/mahamid/Irene/yeast/healthy/180426/004/fas/motl/"
# path_to_motl = "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/006/TM/motl_clean_4b.em"

# path_to_motl = "/struct/mahamid/Irene/yeast/healthy/180426/026/prediction_cnn_IF4_IF8_IF32/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_351.csv"
# output_path = "/struct/mahamid/Irene/yeast/healthy/180426/026/prediction_cnn_IF4_IF8_IF32/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_351.mrc"
# makedirs(name=output_dir, exist_ok=True)
# motl_paths = [
# "/struct/mahamid/Irene/yeast/healthy/180426/006/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/pr_radius_10/undetected/motl_586.csv"
# "/struct/mahamid/Irene/yeast/healthy/180426/004/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_868.csv",
# "/struct/mahamid/Irene/yeast/healthy/180426/005/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_1160.csv",
# "/struct/mahamid/Irene/yeast/healthy/180426/006/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_1166.csv",
# "/struct/mahamid/Irene/yeast/healthy/180426/021/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_1369.csv",
# "/struct/mahamid/Irene/yeast/healthy/180426/024/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_2057.csv",
# "/struct/mahamid/Irene/yeast/healthy/180711/003/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_1839.csv",
# "/struct/mahamid/Irene/yeast/healthy/180711/004/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_447.csv",
# "/struct/mahamid/Irene/yeast/healthy/180711/005/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_398.csv",
# "/struct/mahamid/Irene/yeast/healthy/180711/018/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_980.csv",
# "/struct/mahamid/Irene/yeast/healthy/180713/027/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_362.csv",
# ]

# motl_paths = [
#     # "/struct/mahamid/Irene/yeast/healthy/180426/005/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/motl_2170.csv"
#     # "/struct/mahamid/Irene/yeast/healthy/180426/005/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_1155.csv",
#     # "/struct/mahamid/Irene/yeast/healthy/180426/005/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/detected/motl_1247.csv",
#     # "/struct/mahamid/Irene/yeast/healthy/180426/005/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/checked/additional_tp.csv",
#     # "/struct/mahamid/Irene/yeast/healthy/180426/005/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/checked/fp.csv",
#     "/struct/mahamid/Irene/yeast/healthy/180426/005/prediction_cnn_IF4_IF8_IF32/fas/in_cytosol_mask/motl_83.csv",
#     "/struct/mahamid/Irene/yeast/healthy/180426/005/prediction_cnn_IF4_IF8_IF32/fas/in_cytosol_mask/pr_radius_12/undetected/motl_37.csv",
#     "/struct/mahamid/Irene/yeast/healthy/180426/005/prediction_cnn_IF4_IF8_IF32/fas/in_cytosol_mask/pr_radius_12/detected/motl_47.csv",
#     "/struct/mahamid/Irene/yeast/healthy/180426/005/prediction_cnn_IF4_IF8_IF32/fas/in_cytosol_mask/pr_radius_12/undetected/checked/fp.csv",
#     "/struct/mahamid/Irene/yeast/healthy/180426/005/prediction_cnn_IF4_IF8_IF32/fas/in_cytosol_mask/pr_radius_12/undetected/checked/additional_tp.csv",
# ]
motl_paths = [
    "/struct/mahamid/Irene/yeast/healthy/180426/004/verify_motls/full_motl/no_dimers.csv",
]

z_shift = 0  # shift between original tomogram and subtomogram of analysis
output_shapes = [
    (1000, 928, 960),
    # (1000, 928, 960),
    # (1000, 928, 960),
    # (1000, 928, 960),
    # (1000, 928, 960),
    # (1000, 928, 960),
    # (1000, 928, 960),
    # (1000, 928, 960),
    # (500, 928, 960),
    # (500, 928, 960),
    # (500, 928, 960),
    # (500, 928, 960),
    # (500, 928, 960),
]

assert len(motl_paths) == len(output_shapes)
for path_to_motl, output_shape in zip(motl_paths, output_shapes):
    output_dir, motl_basename = os.path.dirname(path_to_motl), os.path.basename(path_to_motl)
    tomo_name = motl_basename.split(".")[0] + ".mrc"
    # output_path = os.path.join(output_dir[:-51] + "/clean_masks/ribo/TM_added_IF4_IF8_IF32_undetected_1_2_3", tomo_name)
    output_path = os.path.join(output_dir, tomo_name)
    print(path_to_motl)
    print(output_path)
    # os.makedirs(os.path.dirname(output_path), exist_ok=False)
    write_particle_mask_from_motl_in_score_range(path_to_motl=path_to_motl,
                                                 output_path=output_path,
                                                 output_shape=output_shape,
                                                 sphere_radius=8,
                                                 score_range=(0, np.inf),
                                                 number_of_particles=None,
                                                 z_shift=z_shift,
                                                 particles_in_tom_format=True)
