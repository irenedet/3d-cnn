import os
import time

import numpy as np

from file_actions.writers.csv import motl_writer
from tomogram_utils.peak_toolbox.utils import read_motl_coordinates_and_values


def get_tomo_motl_paths(cnn_predictions_dir, motl_endings):
    tomo_motl_paths = []
    cnn_predictions_list = os.listdir(cnn_predictions_dir)
    for cnn_dir in cnn_predictions_list:
        # if len(cnn_dir) > 0:
        # if "3rd" not in cnn_dir:
        # if "3rd_round" in cnn_dir:
        if cnn_dir in [
            "ribo_fas_defocus_spombe_IF4",
            "ribo_fas_defocus_spombe_IF8",
            "ribo_fas_defocus_spombe_IF32",
            "ribo_fas_vpp_spombe_IF4",
            "ribo_fas_vpp_spombe_IF8",
            "ribo_fas_vpp_spombe_IF32",
        ]:
            motl_dir = os.path.join(cnn_predictions_dir, cnn_dir)
            motl_dir = os.path.join(motl_dir, tomo_name)
            for motl_ending in motl_endings:
                motl_dir_ = os.path.join(motl_dir, motl_ending)
                # print(motl_dir)
                if os.path.isdir(motl_dir_):
                    files = os.listdir(motl_dir_)
                    for file in files:
                        if "motl" == file[:4] and ".csv" == file[-4:]:
                            motl_file = os.path.join(motl_dir_, file)
                            tomo_motl_paths.append(motl_file)
    return tomo_motl_paths


def filter_coords(tomo_motl_paths, radius, value_threshold):
    total_coordinates = []
    total_values = []
    filtered_coordinates = []
    filtered_values = []

    for index, motl_path in enumerate(tomo_motl_paths):
        motl_values, motl_coords = read_motl_coordinates_and_values(path_to_motl=motl_path)
        # if index == 0:
        #     filtered_coordinates = list(motl_coords)
        #     filtered_values = list(motl_values)
        # else:
        total_coordinates += list(motl_coords)
        total_values += list(motl_values)

    assert len(total_values) == len(total_coordinates)
    start = time.time()
    for new_point, new_value in zip(total_coordinates, total_values):
        flag = "in"
        # if new_value > value_threshold:
        for point in filtered_coordinates:
            difference = point - new_point
            distance = np.linalg.norm(difference)
            if distance <= radius:
                flag = "out"
        if flag == "in":
            filtered_coordinates.append(new_point)
            filtered_values.append(new_value)
    end = time.time()
    print("time elapsed:", end - start, "secs.")
    print("len(filtered_coordinates) =", len(filtered_coordinates))
    return filtered_coordinates, filtered_values


def write_combined_list(path_to_output, filtered_coordinates, filtered_values):
    os.makedirs(path_to_output, exist_ok=True)
    # remove old paths to avoid confusions:
    for file in os.listdir(path_to_output):
        if "motl" == file[:4]:
            file_path = os.path.join(path_to_output, file)
            new_path = os.path.join(path_to_output, "old_" + file)
            os.rename(file_path, new_path)
            print("File removed:", file_path)
    n = len(filtered_coordinates)
    output_motl_name = "motl_" + str(n) + ".csv"

    motl_writer(path_to_output_folder=path_to_output,
                list_of_peak_coords=filtered_coordinates,
                list_of_peak_scores=filtered_values,
                in_tom_format=True,
                order_by_score=False, list_of_angles=False,
                motl_name=output_motl_name)


pr_radius_dict = {"ribo": "10", "fas": "12"}
sample_type = "healthy"
particle = "ribo"
pr_radius = pr_radius_dict[particle]
radius = 18  # to avoid dimers
value_threshold = 0
conditions = "vpp"
motl_type = "undetected"  # detected, undetected, or full

if conditions == "vpp":
    tomo_names = [
        "180426/004",
        "180426/005",
        "180426/021",
        "180426/024",
        "180711/003",
        "180711/004",
        "180711/005",
        "180711/018",
        "180713/027",
        "180426/006",
    ]
    # VPP DATA:
    cnn_predictions_dirs = [
        "/struct/mahamid/Irene/segmentation_3dunets/vpp_out/predictions",
        "/struct/mahamid/Irene/3dunet_cv/alex_filter/singleclass/ribos/vpp/out/predictions"
    ]

else:
    # DEFOCUS DATA:
    tomo_names = [
        "180426/026",
        "180426/027",
        "180426/028",
        "180426/029",
        "180426/030",
        "180426/034",
        "180426/037",
        "180426/041",
        "180426/043",
        "180426/045",
    ]
    cnn_predictions_dirs = [
        "/struct/mahamid/Irene/segmentation_3dunets/def_out/predictions",
        "/struct/mahamid/Irene/3dunet_cv/alex_filter/singleclass/ribos/defocus/out/predictions"
    ]


def get_motl_dir1_undetected(particle, pr_radius):
    motl_ending1 = particle + "/in_cytosol_mask/pr_radius_" + pr_radius + "/undetected"
    return motl_ending1


def get_motl_dir2_undetected(particle, pr_radius):
    motl_dir2 = particle + "/pr_radius_" + pr_radius + "/undetected"
    return motl_dir2


def get_motl_dir1_detected(particle, pr_radius):
    motl_ending1 = particle + "/in_cytosol_mask/pr_radius_" + pr_radius + "/detected"
    return motl_ending1


def get_motl_dir2_detected(particle, pr_radius):
    motl_dir2 = particle + "/pr_radius_" + pr_radius + "/detected"
    return motl_dir2


def get_motl_dir1_full(particle):
    motl_ending1 = particle + "/in_cytosol_mask"
    return motl_ending1


def get_motl_dir2_full(particle):
    motl_dir2 = particle
    return motl_dir2


# motl_dirX = os.path.join(motl_dir, particle)
# motl_dir0 = os.path.join(motl_dir, "ribo/in_lamella_file/pr_radius_10/undetected")
if motl_type == "undetected":
    motl_ending1 = get_motl_dir1_undetected(particle, pr_radius)
    motl_ending2 = get_motl_dir2_undetected(particle, pr_radius)
elif motl_type == "detected":
    motl_ending1 = get_motl_dir1_detected(particle, pr_radius)
    motl_ending2 = get_motl_dir2_detected(particle, pr_radius)
elif motl_type == "full":
    motl_ending1 = get_motl_dir1_full(particle)
    motl_ending2 = get_motl_dir2_full(particle)

motl_endings = [motl_ending1, motl_ending2]


# Generate output dir with name related to IF4, IF8, and IF32
def get_path_to_output_undetected(tomo_name, particle, sample_type):
    output_subfolder = "prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_def_round/" + particle + "/in_cytosol_mask/pr_radius_" + pr_radius + "/undetected"
    if "ScED" in tomo_name:
        path_to_output = os.path.join("/struct/mahamid/Irene/yeast", tomo_name)
    else:
        if sample_type == "ED":
            path_to_output = os.path.join("/struct/mahamid/Irene/yeast/ED", tomo_name)
        else:
            path_to_output = os.path.join("/struct/mahamid/Irene/yeast/healthy", tomo_name)

    path_to_output = os.path.join(path_to_output, output_subfolder)
    return path_to_output


def get_path_to_output_detected(tomo_name, particle, sample_type):
    output_subfolder = "prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_def_round/" + particle + "/in_cytosol_mask/pr_radius_" + pr_radius + "/detected"
    if "ScED" in tomo_name:
        path_to_output = os.path.join("/struct/mahamid/Irene/yeast", tomo_name)
    else:
        if sample_type == "ED":
            path_to_output = os.path.join("/struct/mahamid/Irene/yeast/ED", tomo_name)
        else:
            path_to_output = os.path.join("/struct/mahamid/Irene/yeast/healthy", tomo_name)

    path_to_output = os.path.join(path_to_output, output_subfolder)
    return path_to_output


def get_path_to_output_full(tomo_name, particle, sample_type):
    output_subfolder = "prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_def_round/" + particle + "/in_cytosol_mask"
    if "ScED" in tomo_name:
        path_to_output = os.path.join("/struct/mahamid/Irene/yeast", tomo_name)
    else:
        if sample_type == "ED":
            path_to_output = os.path.join("/struct/mahamid/Irene/yeast/ED", tomo_name)
        else:
            path_to_output = os.path.join("/struct/mahamid/Irene/yeast/healthy", tomo_name)

    path_to_output = os.path.join(path_to_output, output_subfolder)
    return path_to_output


for tomo_name in tomo_names:
    print("tomo_name =", tomo_name)
    tomo_motl_paths = []
    full_names = []
    for cnn_predictions_dir in cnn_predictions_dirs:
        tomo_motl_paths += get_tomo_motl_paths(cnn_predictions_dir, motl_endings)

    print("MOTLS TO COMBINE:", tomo_motl_paths)
    # assert len(tomo_motl_paths) == 3
    filtered_coordinates, filtered_values = filter_coords(tomo_motl_paths=tomo_motl_paths, radius=radius,
                                                          value_threshold=value_threshold)
    if motl_type == "undetected":
        path_to_output = get_path_to_output_undetected(tomo_name, particle, sample_type)
    elif motl_type == "detected":
        path_to_output = get_path_to_output_detected(tomo_name, particle, sample_type)
    elif motl_type == "full":
        path_to_output = get_path_to_output_full(tomo_name, particle, sample_type)
    # print(path_to_output)
    write_combined_list(path_to_output, filtered_coordinates, filtered_values)

# clean_motls = [
#     "/struct/mahamid/Irene/yeast/healthy/180426/004/ribos/motl/corrected_motl_verified_shifted_to_origin.csv",
#     "/struct/mahamid/Irene/yeast/healthy/180426/005/motl/corrected_motl.csv",
#     "/struct/mahamid/Irene/yeast/healthy/180426/006/motl_clean_nonova_4b_shifted.csv",
#     "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/021/TM/motl_clean_4b.em",
#     "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/024/TM/motl_clean_4b.em",
#     "/struct/mahamid/Irene/yeast/healthy/180711/003/ribos/motl/corrected_motl_verified.csv",
#     "/struct/mahamid/Irene/yeast/healthy/180711/004/ribos/motl/corrected_motl_verified.csv",
#     "/struct/mahamid/Irene/yeast/healthy/180711/005/ribos/motl/corrected_motl_verified.csv",
#     "/struct/mahamid/Irene/yeast/healthy/180711/018/ribos/motl/corrected_motl_verified.csv",
#     "/struct/mahamid/Irene/yeast/healthy/180713/027/ribos/motl/corrected_motl.csv",
# ]
#
# assert len(tomo_names) == len(clean_motls)
# # Generate output dir with name related to IF4, IF8, and IF32
# def get_path_to_clean_output(tomo_name, particle, sample_type):
#     output_subfolder = "clean_motls/" + particle + "/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_3_def_rounds"
#     if "ScED" in tomo_name:
#         path_to_output = os.path.join("/struct/mahamid/Irene/yeast", tomo_name)
#     else:
#         if sample_type == "ED":
#             path_to_output = os.path.join("/struct/mahamid/Irene/yeast/ED", tomo_name)
#         else:
#             path_to_output = os.path.join("/struct/mahamid/Irene/yeast/healthy", tomo_name)
#
#     path_to_output = os.path.join(path_to_output, output_subfolder)
#     return path_to_output
#
#
# for tomo_name, pre_clean_motl in zip(tomo_names, clean_motls):
#     extra_motl_dir = os.path.join("/struct/mahamid/Irene/yeast/healthy", tomo_name)
#     extra_motl_dir = os.path.join(extra_motl_dir,
#                                   "prediction_cnn_IF4_IF8_IF32_3nd_round/ribo/in_cytosol_mask/pr_radius_10/undetected")
#     tomo_motl_paths = [pre_clean_motl]
#     for file in os.listdir(extra_motl_dir):
#         if file[:4] == "motl" and file.split(".")[1] == "csv":
#             print(file)
#             extra_motl_path = os.path.join(extra_motl_dir, file)
#             tomo_motl_paths.append(extra_motl_path)
#     filtered_coordinates, filtered_values = filter_coords(tomo_motl_paths=tomo_motl_paths, radius=radius,
#                                          value_threshold=value_threshold)
#     path_to_output = get_path_to_clean_output(tomo_name, particle, sample_type)
#     print(path_to_output)
#     print("PARTICLES:", len(filtered_coordinates))
#     write_combined_list(path_to_output, filtered_coordinates, filtered_values)


#
# motls_ribo = {"180426/004":
#                   {"manual": "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/004/eman/ribo_4b_manual.txt",
#                    "TM": "/struct/mahamid/Irene/yeast/healthy/180426/004/verify_motls/old_motl_clean_4b.csv",
#                    "round_1_vpp": "/struct/mahamid/Irene/yeast/healthy/180426/004/prediction_cnn_IF4_IF8_IF32/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_836.csv",
#                    "round_1_vpp_round_1_def": "nan",
#                    "round_1_vpp_round_1_2_def": "/struct/mahamid/Irene/yeast/healthy/180426/004/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_866.csv",
#                    "old_CNN": "/struct/mahamid/Irene/yeast/healthy/180426/004/verify_motls/motl_482_checked.csv"},
#               "180426/005":
#                   ["/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/005/eman/ribo_4b_manual.txt",
#                    "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/004/TM/motl_clean_4b.em",
#                    "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/005/eman/ribo_CNN_undetected_4b_checked.txt",
#                    "/struct/mahamid/Irene/yeast/healthy/180426/005/verify_motls/motl_307_checked.csv"],
#               "180426/006":
#                   ["/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/006/eman/ribo_4b_manual.txt",
#                    "/struct/mahamid/Irene/yeast/healthy/180426/006/clean_motls/ribo/TM/ribo_4b_shifted-16.csv",
#                    "/struct/mahamid/Irene/yeast/healthy/180426/006/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_1150.csv"],
#               "180426/021":
#                   ["/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/021/eman/ribo_4b_manual.txt",
#                    "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/021/TM/motl_clean_4b.em",
#                    "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/021/eman/ribo_CNN_undetected_4b_checked.txt"],
#               "180426/024":
#                   ["/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/024/eman/ribo_4b_manual.txt",
#                    "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/024/TM/motl_clean_4b.em",
#                    "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/024/eman/ribo_CNN_undetected_4b_checked.txt"],
#               "180711/003":
#                   ["/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/003/eman/ribo_4b_manual.txt",
#                    "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/003/TM/motl_clean_4b.em",
#                    "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/003/eman/ribo_CNN_undetected_4b_checked.txt",
#                    "/struct/mahamid/Irene/predictions/180711/003/ribo/motl_2521_checked_shifted.csv"],
#               "180711/004":
#                   ["/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/004/eman/ribo_4b_manual.txt",
#                    "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/004/TM/motl_clean_4b.em",
#                    "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/004/eman/ribo_CNN_undetected_4b_checked.txt",
#                    "/struct/mahamid/Shared/For_Irene/predictions/180711/004/motl_587_checked.csv"],
#               "180711/005":
#                   ["/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/005/eman/ribo_4b_manual.txt",
#                    "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/005/TM/motl_clean_4b.em",
#                    "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/005/eman/ribo_CNN_undetected_4b_checked.txt",
#                    "/struct/mahamid/Irene/predictions/180711/005/ribo/motl_631_checked_shifted.csv"],
#               "180711/018":
#                   ["/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/018/eman/ribo_4b_manual.txt",
#                    "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/018/TM/motl_clean_4b.em",
#                    "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/018/eman/ribo_CNN_undetected_4b_checked.txt",
#                    "/struct/mahamid/Shared/For_Irene/predictions/180711/018/motl_459_checked.csv"],
#               "180713/027":
#                   ["/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180713/027/eman/ribo_4b_manual.txt",
#                    "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180713/027/TM/motl_clean_4b.em",
#                    "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180713/027/eman/ribo_CNN_undetected_4b_checked.txt",
#                    "/struct/mahamid/Irene/predictions/180713/027/ribo/motl_885_thr.csv"],
#               }

extra = {
    "180426/004": "/struct/mahamid/Irene/yeast/healthy/180426/004/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_866.csv",
    "180426/005": "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/005/eman/ribo_CNN_undetected_4b_checked.txt",
    "180426/006": "/struct/mahamid/Irene/yeast/healthy/180426/006/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_1150.csv",
    "180426/021": "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/021/eman/ribo_CNN_undetected_4b_checked.txt",
    "180426/024": "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/024/eman/ribo_CNN_undetected_4b_checked.txt",
    "180711/003": "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/003/eman/ribo_CNN_undetected_4b_checked.txt",
    "180711/004": "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/004/eman/ribo_CNN_undetected_4b_checked.txt",
    "180711/005": "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/005/eman/ribo_CNN_undetected_4b_checked.txt",
    "180711/018": "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/018/eman/ribo_CNN_undetected_4b_checked.txt",
    "180713/027": "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180713/027/eman/ribo_CNN_undetected_4b_checked.txt",
}

# no_dimer_lists = {
#     "180426/004": "/struct/mahamid/Irene/yeast/healthy/180426/004/verify_motls/full_motl/no_dimers_radius15.csv",
#     "180426/005": "/struct/mahamid/Irene/yeast/healthy/180426/005/verify_motls/full_motl/no_dimers_radius15.csv",
#     "180426/006": "/struct/mahamid/Irene/yeast/healthy/180426/006/verify_motls/full_motl/no_dimers_radius15.csv",
#     "180426/021": "/struct/mahamid/Irene/yeast/healthy/180426/021/verify_motls/full_motl/no_dimers_radius15.csv",
#     "180426/024": "/struct/mahamid/Irene/yeast/healthy/180426/024/verify_motls/full_motl/no_dimers_radius15.csv",
#     "180711/003": "/struct/mahamid/Irene/yeast/healthy/180711/003/verify_motls/full_motl/no_dimers_radius15.csv",
#     "180711/004": "/struct/mahamid/Irene/yeast/healthy/180711/004/verify_motls/full_motl/no_dimers_radius15.csv",
#     "180711/005": "/struct/mahamid/Irene/yeast/healthy/180711/005/verify_motls/full_motl/no_dimers_radius15.csv",
#     "180711/018": "/struct/mahamid/Irene/yeast/healthy/180711/018/verify_motls/full_motl/no_dimers_radius15.csv",
#     "180713/027": "/struct/mahamid/Irene/yeast/healthy/180713/027/verify_motls/full_motl/no_dimers_radius15.csv",
# }
# manual = {
#     "180426/004": "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/004/eman/ribo_4b_manual.txt",
#     "180426/005": "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/005/eman/ribo_4b_manual.txt",
#     "180426/006": "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/006/eman/ribo_4b_manual.txt",
#     "180426/021": "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/021/eman/ribo_4b_manual.txt",
#     "180426/024": "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/024/eman/ribo_4b_manual.txt",
#     "180711/003": "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/003/eman/ribo_4b_manual.txt",
#     "180711/004": "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/004/eman/ribo_4b_manual.txt",
#     "180711/005": "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/005/eman/ribo_4b_manual.txt",
#     "180711/018": "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/018/eman/ribo_4b_manual.txt",
#     "180713/027": "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180713/027/eman/ribo_4b_manual.txt",
# }
#
# TM = {
#     "180426/004": "/struct/mahamid/Irene/yeast/healthy/180426/004/verify_motls/old_motl_clean_4b.csv",
#     "180426/005": "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/004/TM/motl_clean_4b.em",
#     "180426/006": "/struct/mahamid/Irene/yeast/healthy/180426/006/clean_motls/ribo/TM/ribo_4b_shifted-16.csv",
#     "180426/021": "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/021/TM/motl_clean_4b.em",
#     "180426/024": "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/024/TM/motl_clean_4b.em",
#     "180711/003": "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/003/TM/motl_clean_4b.em",
#     "180711/004": "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/004/TM/motl_clean_4b.em",
#     "180711/005": "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/005/TM/motl_clean_4b.em",
#     "180711/018": "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/018/TM/motl_clean_4b.em",
#     "180713/027": "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180713/027/TM/motl_clean_4b.em",
# }
# round_1vpp = {
#     "180426/004": "/struct/mahamid/Irene/yeast/healthy/180426/004/prediction_cnn_IF4_IF8_IF32/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_836.csv",
#     "180426/005": "/struct/mahamid/Irene/yeast/healthy/180426/005/prediction_cnn_IF4_IF8_IF32/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_864.csv",
#     "180426/006": "/struct/mahamid/Irene/yeast/healthy/180426/006/prediction_cnn_IF4_IF8_IF32/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_1155.csv",
#     "180426/021": "/struct/mahamid/Irene/yeast/healthy/180426/021/prediction_cnn_IF4_IF8_IF32/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_1098.csv",
#     "180426/024": "/struct/mahamid/Irene/yeast/healthy/180426/024/prediction_cnn_IF4_IF8_IF32/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_1815.csv",
#     "180711/003": "/struct/mahamid/Irene/yeast/healthy/180711/003/prediction_cnn_IF4_IF8_IF32/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_1632.csv",
#     "180711/004": "/struct/mahamid/Irene/yeast/healthy/180711/004/prediction_cnn_IF4_IF8_IF32/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_273.csv",
#     "180711/005": "/struct/mahamid/Irene/yeast/healthy/180711/005/prediction_cnn_IF4_IF8_IF32/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_300.csv",
#     "180711/018": "/struct/mahamid/Irene/yeast/healthy/180711/018/prediction_cnn_IF4_IF8_IF32/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_949.csv",
#     "180713/027": "/struct/mahamid/Irene/yeast/healthy/180713/027/prediction_cnn_IF4_IF8_IF32/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_126.csv",
# }
#
# round_1vpp_1def = {
#     "180426/004": "/struct/mahamid/Irene/yeast/healthy/180426/004/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_def_round/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_739.csv",
#     "180426/005": "/struct/mahamid/Irene/yeast/healthy/180426/005/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_def_round/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_866.csv",
#     "180426/006": "/struct/mahamid/Irene/yeast/healthy/180426/006/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_def_round/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_1050.csv",
#     "180426/021": "/struct/mahamid/Irene/yeast/healthy/180426/021/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_def_round/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_1091.csv",
#     "180426/024": "/struct/mahamid/Irene/yeast/healthy/180426/024/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_def_round/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_1415.csv",
#     "180711/003": "/struct/mahamid/Irene/yeast/healthy/180711/003/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_def_round/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_1568.csv",
#     "180711/004": "/struct/mahamid/Irene/yeast/healthy/180711/004/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_def_round/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_318.csv",
#     "180711/005": "/struct/mahamid/Irene/yeast/healthy/180711/005/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_def_round/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_269.csv",
#     "180711/018": "/struct/mahamid/Irene/yeast/healthy/180711/018/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_def_round/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_945.csv",
#     "180713/027": "/struct/mahamid/Irene/yeast/healthy/180713/027/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_def_round/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_239.csv",
# }
#
# round_1vpp_1_2def = {
#     "180426/004": "/struct/mahamid/Irene/yeast/healthy/180426/004/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_866.csv",
#     "180426/005": "/struct/mahamid/Irene/yeast/healthy/180426/005/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_1155.csv",
#     "180426/006": "/struct/mahamid/Irene/yeast/healthy/180426/006/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_1150.csv",
#     "180426/021": "/struct/mahamid/Irene/yeast/healthy/180426/021/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_1364.csv",
#     "180426/024": "/struct/mahamid/Irene/yeast/healthy/180426/024/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_2047.csv",
#     "180711/003": "/struct/mahamid/Irene/yeast/healthy/180711/003/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_1829.csv",
#     "180711/004": "/struct/mahamid/Irene/yeast/healthy/180711/004/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_445.csv",
#     "180711/005": "/struct/mahamid/Irene/yeast/healthy/180711/005/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_396.csv",
#     "180711/018": "/struct/mahamid/Irene/yeast/healthy/180711/018/prediction_cnn_IF4_IF8_IF32_1_vpp_round_1_2_def_rounds/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_968.csv",
#     "180713/027": "/struct/mahamid/Irene/yeast/healthy/180713/027/prediction_cnn_IF4_IF8_IF32/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_126.csv",
# }
#
# old_CNN = {
#     "180426/004": "/struct/mahamid/Irene/yeast/healthy/180426/004/verify_motls/motl_482_checked.csv",
#     "180426/005": "/struct/mahamid/Irene/yeast/healthy/180426/005/verify_motls/motl_307_checked.csv",
#     "180426/006": "",
#     "180426/021": "",
#     "180426/024": "",
#     "180711/003": "/struct/mahamid/Irene/predictions/180711/003/ribo/motl_2521_checked_shifted.csv",
#     "180711/004": "/struct/mahamid/Shared/For_Irene/predictions/180711/004/motl_587_checked.csv",
#     "180711/005": "/struct/mahamid/Irene/predictions/180711/005/ribo/motl_631_checked_shifted.csv",
#     "180711/018": "/struct/mahamid/Shared/For_Irene/predictions/180711/018/motl_459_checked.csv",
#     "180713/027": "/struct/mahamid/Irene/predictions/180713/027/ribo/motl_885_thr.csv",
# }
