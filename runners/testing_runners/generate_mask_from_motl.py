import os
import os.path

import numpy as np
import pandas as pd

from file_actions.readers.em import read_em
from file_actions.readers.motl import read_motl_from_csv
from file_actions.writers.mrc import write_mrc_dataset
# from file_actions.writers.h5 import write_particle_mask_from_motl_in_score_range
from tomogram_utils.coordinates_toolbox.utils import \
    extract_coordinates_from_em_motl
from tomogram_utils.peak_toolbox.utils import paste_sphere_in_dataset


def read_txt_list(txt_path):
    txt_list_df = pd.read_csv(txt_path, sep="\t", names=["x", "y", "z"])
    return txt_list_df


def generate_particle_mask_from_motl(path_to_motl: str, output_shape: tuple,
                                     sphere_radius: int, value: int = 1) -> np.array:
    motl_extension = os.path.basename(path_to_motl).split(".")[-1]
    # print("motl_extension", motl_extension)
    assert motl_extension in ["csv", "em", "txt"]

    if motl_extension == "csv":
        motive_list = read_motl_from_csv(path_to_motl)
        coordinates = [
            np.array([int(row[9]), int(row[8]), int(row[7])])
            for
            row in motive_list]

    elif motl_extension == "em":
        _, motive_list = read_em(path_to_emfile=path_to_motl)
        coordinates = extract_coordinates_from_em_motl(motive_list)
        coordinates = [[int(p[2]), int(p[1]), int(p[0])] for p
                       in coordinates]
    else:
        motive_list = read_txt_list(txt_path=path_to_motl)
        coordinates = motive_list[["x", "y", "z"]].values
        coordinates = [[int(p[2]), int(p[1]), int(p[0])] for p
                       in coordinates]

    predicted_dataset = np.zeros(output_shape)

    for center in coordinates:
        paste_sphere_in_dataset(dataset=predicted_dataset, center=center,
                                radius=sphere_radius, value=value)

    return predicted_dataset


# motls_ribo = {
# "180426/004":
#               ["/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/004/eman/ribo_4b_manual.txt",
#                "/struct/mahamid/Irene/yeast/healthy/180426/004/motl/old_motl_clean_4b.em",
#                "/struct/mahamid/Irene/yeast/healthy/180426/004/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_866.csv",
#                "/struct/mahamid/Shared/For_Irene/predictions/180426/004/motl_482_checked.csv"],
#           "180426/005":
#               ["/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/005/eman/ribo_4b_manual.txt",
#                "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/004/TM/motl_clean_4b.em",
#                "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/005/eman/ribo_CNN_undetected_4b_checked.txt",
#                "/struct/mahamid/Shared/For_Irene/predictions/180426/005/motl_307_checked.csv"],
#           "180426/006":
#               ["/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/006/eman/ribo_4b_manual.txt",
#                "/struct/mahamid/Irene/yeast/healthy/180426/006/clean_motls/ribo/TM/ribo_4b_shifted-16.csv",
#                "/struct/mahamid/Irene/yeast/healthy/180426/006/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_1150.csv"],
#           "180426/021":
#               ["/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/021/eman/ribo_4b_manual.txt",
#                "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/021/TM/motl_clean_4b.em",
#                "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/021/eman/ribo_CNN_undetected_4b_checked.txt"],
#           "180426/024":
#               ["/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/024/eman/ribo_4b_manual.txt",
#                "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/024/TM/motl_clean_4b.em",
#                "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/024/eman/ribo_CNN_undetected_4b_checked.txt"],
#           "180711/003":
#               ["/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/003/eman/ribo_4b_manual.txt",
#                "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/003/TM/motl_clean_4b.em",
#                "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/003/eman/ribo_CNN_undetected_4b_checked.txt",
#                "/struct/mahamid/Irene/predictions/180711/003/ribo/motl_2521_checked_shifted.csv"],
#           "180711/004":
#               ["/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/004/eman/ribo_4b_manual.txt",
#                "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/004/TM/motl_clean_4b.em",
#                "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/004/eman/ribo_CNN_undetected_4b_checked.txt",
#                "/struct/mahamid/Shared/For_Irene/predictions/180711/004/motl_587_checked.csv"],
#           "180711/005":
#               ["/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/005/eman/ribo_4b_manual.txt",
#                "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/005/TM/motl_clean_4b.em",
#                "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/005/eman/ribo_CNN_undetected_4b_checked.txt",
#                "/struct/mahamid/Irene/predictions/180711/005/ribo/motl_631_checked_shifted.csv"],
#           "180711/018":
#               ["/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/018/eman/ribo_4b_manual.txt",
#                "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/018/TM/motl_clean_4b.em",
#                "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/018/eman/ribo_CNN_undetected_4b_checked.txt",
#                "/struct/mahamid/Shared/For_Irene/predictions/180711/018/motl_459_checked.csv"],
#           "180713/027":
#               ["/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180713/027/eman/ribo_4b_manual.txt",
#                "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180713/027/TM/motl_clean_4b.em",
#                "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180713/027/eman/ribo_CNN_undetected_4b_checked.txt",
#                "/struct/mahamid/Irene/predictions/180713/027/ribo/motl_885_thr.csv"],
#           }
# motls_ribo = {
#     "180426/004": ["/struct/mahamid/Irene/yeast/healthy/180426/004/verify_motls/full_motl/no_dimers_radius15.csv"],
#     "180426/005": ["/struct/mahamid/Irene/yeast/healthy/180426/005/verify_motls/full_motl/no_dimers_radius15.csv"],
#     "180426/006": ["/struct/mahamid/Irene/yeast/healthy/180426/006/verify_motls/full_motl/no_dimers_radius15.csv"],
#     "180426/021": ["/struct/mahamid/Irene/yeast/healthy/180426/021/verify_motls/full_motl/no_dimers_radius15.csv"],
#     "180426/024": ["/struct/mahamid/Irene/yeast/healthy/180426/024/verify_motls/full_motl/no_dimers_radius15.csv"],
#     "180711/003": ["/struct/mahamid/Irene/yeast/healthy/180711/003/verify_motls/full_motl/no_dimers_radius15.csv"],
#     "180711/004": ["/struct/mahamid/Irene/yeast/healthy/180711/004/verify_motls/full_motl/no_dimers_radius15.csv"],
#     "180711/005": ["/struct/mahamid/Irene/yeast/healthy/180711/005/verify_motls/full_motl/no_dimers_radius15.csv"],
#     "180711/018": ["/struct/mahamid/Irene/yeast/healthy/180711/018/verify_motls/full_motl/no_dimers_radius15.csv"],
#     "180713/027": ["/struct/mahamid/Irene/yeast/healthy/180713/027/verify_motls/full_motl/no_dimers_radius15.csv"],
# }
motls_ribo = {
    # "180426/004": ["/struct/mahamid/Irene/yeast/healthy/180426/004/verify_motls/full_motl/no_dimers_radius15_in_cyto.csv"],
    "180426/005": ["/struct/mahamid/Irene/yeast/healthy/180426/005/verify_motls/full_motl/no_dimers_radius15_in_cyto.csv"],
    "180426/006": ["/struct/mahamid/Irene/yeast/healthy/180426/006/verify_motls/full_motl/no_dimers_radius15_in_cyto.csv"],
    "180426/021": ["/struct/mahamid/Irene/yeast/healthy/180426/021/verify_motls/full_motl/no_dimers_radius15_in_cyto.csv"],
    "180426/024": ["/struct/mahamid/Irene/yeast/healthy/180426/024/verify_motls/full_motl/no_dimers_radius15_in_cyto.csv"],
    "180711/003": ["/struct/mahamid/Irene/yeast/healthy/180711/003/verify_motls/full_motl/no_dimers_radius15_in_cyto.csv"],
    "180711/004": ["/struct/mahamid/Irene/yeast/healthy/180711/004/verify_motls/full_motl/no_dimers_radius15_in_cyto.csv"],
    "180711/005": ["/struct/mahamid/Irene/yeast/healthy/180711/005/verify_motls/full_motl/no_dimers_radius15_in_cyto.csv"],
    "180711/018": ["/struct/mahamid/Irene/yeast/healthy/180711/018/verify_motls/full_motl/no_dimers_radius15_in_cyto.csv"],
    "180713/027": ["/struct/mahamid/Irene/yeast/healthy/180713/027/verify_motls/full_motl/no_dimers_radius15_in_cyto.csv"],
}
# output_shape = (1000, 928, 960)
output_shapes = {
    "180426/004": (960, 928, 1000),
    "180426/005": (960, 928, 1000),
    "180426/006": (960, 928, 1000),
    "180426/021": (959, 927, 1000),
    "180426/024": (959, 927, 1000),
    "180711/003": (960, 928, 500),
    "180711/004": (960, 928, 500),
    "180711/005": (960, 928, 500),
    "180711/018": (960, 928, 500),
    "180713/027": (960, 928, 500),
}
for tomo_name in list(motls_ribo.keys()):
    motl_paths = motls_ribo[tomo_name]
    shx, shy, shz = output_shapes[tomo_name]
    output_shape = (shz, shy, shx)
    output_dir = "/struct/mahamid/Irene/yeast/healthy/" + tomo_name + "/verify_motls/full_motl/"
    os.makedirs(output_dir, exist_ok=True)

    for path_to_motl in motl_paths:
        motl_basename = os.path.basename(path_to_motl)
        tomo_name = motl_basename.split(".")[0] + ".mrc"
        output_path = os.path.join(output_dir, tomo_name)
        print(path_to_motl)
        print(output_path)
        dataset = generate_particle_mask_from_motl(path_to_motl=path_to_motl, output_shape=output_shape,
                                                   sphere_radius=8, value=1)

        write_mrc_dataset(mrc_path=output_path, array=dataset)

