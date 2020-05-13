import os
import time

import numpy as np

from file_actions.writers.csv import motl_writer
from tomogram_utils.peak_toolbox.utils import read_motl_coordinates_and_values


def get_motl_maths(tomo_name: str) -> list:
    motl_paths = []

    TM_motl = os.path.join("/struct/mahamid/twalther/Processing", tomo_name)
    TM_motl = os.path.join(TM_motl, "TM/motl_clean_4b.em")
    motl_paths.append(TM_motl)

    for cnn_dir in cnn_predictions_list:
        CNN_label_dir = os.path.join(CNN_predictions_dir, cnn_dir)
        CNN_label_dir = os.path.join(CNN_label_dir, tomo_name)
        exists_pred = os.path.isdir(CNN_label_dir)
        if exists_pred:
            motl_dir = os.path.join(CNN_label_dir, 'ribo/in_lamella_file/pr_radius_10/undetected')

            for file_basename in os.listdir(motl_dir):
                if "motl" in file_basename:
                    motl_file = os.path.join(motl_dir, file_basename)
                    motl_paths.append(motl_file)
    return motl_paths


if __name__ == "__main__":
    # missing lists
    # 190301/004
    # 190301/007
    # 190301/014
    # 190301/017
    # 190301/018
    # 190301/019
    # "/struct/mahamid/twalther/Processing/190301/004/TM/motl_clean_4b.em",
    # "/struct/mahamid/twalther/Processing/190301/007/TM/motl_clean_4b.em",
    # "/struct/mahamid/twalther/Processing/190301/014/TM/motl_clean_4b.em",
    # "/struct/mahamid/twalther/Processing/190301/017/TM/motl_clean_4b.em",
    # "/struct/mahamid/twalther/Processing/190301/018/TM/motl_clean_4b.em",
    # "/struct/mahamid/twalther/Processing/190301/019/TM/motl_clean_4b.em",

    tomo_names = [
        "190301/001",
        "190301/002",
        "190301/006",
        "190301/010",
        "190301/011",
        "190301/013",
        "190301/015",
        "190301/020",
        "190301/021",
        "190301/026",
        "190301/029",
        "190301/030",
        "190301/004",
        "190301/007",
        "190301/014",
        "190301/017",
        "190301/018",
        "190301/019",
    ]


    CNN_predictions_dir = "/struct/mahamid/Irene/scratch/trueba/3d-cnn/cnn_evaluation/ribo_sph_masks"
    cnn_predictions_list = os.listdir(CNN_predictions_dir)
    radius = 10
    value_threshold = 1200
    full_names = []
    for tomo_name in tomo_names:
        print("tomo_name =", tomo_name)
        tomo_motl_paths = get_motl_maths(tomo_name)
        print(tomo_motl_paths)
        total_coordinates = []
        total_values = []
        filtered_coordinates = []
        for index, motl_path in enumerate(tomo_motl_paths):
            motl_values, motl_coords = read_motl_coordinates_and_values(path_to_motl=motl_path)
            if index == 0:
                print("total points from TM clean =", len(list(motl_values)))
                filtered_coordinates = list(motl_coords)
            else:
                total_coordinates += list(motl_coords)[:1000]
                total_values += list(motl_values)[:1000]


        start = time.time()
        for new_point, new_value in zip(total_coordinates, total_values):
            flag = "in"
            if new_value > value_threshold:
                for point in filtered_coordinates:
                    difference = point - new_point
                    distance = np.linalg.norm(difference)
                    if distance <= radius:
                        flag = "out"
                if flag == "in":
                    filtered_coordinates.append(new_point)
        end = time.time()
        print("time elapsed:", end - start, "secs.")
        print("len(filtered_coordinates) =", len(filtered_coordinates))

        path_to_output = os.path.join("/struct/mahamid/Irene/yeast/ED", tomo_name)
        path_to_output = os.path.join(path_to_output, "clean_motls/ribo/combined")
        os.makedirs(path_to_output, exist_ok=True)

        n = len(filtered_coordinates)
        filtered_values = list(np.ones(n))
        output_motl_name = "TM_cnnIF4_cnnIF8_cnnIF32_motl_" + str(n) + ".csv"

        motl_writer(path_to_output_folder=path_to_output,
                    list_of_peak_coords=filtered_coordinates,
                    list_of_peak_scores=filtered_values,
                    in_tom_format=True,
                    order_by_score=False, list_of_angles=False,
                    motl_name=output_motl_name)
        full_names.append(os.path.join(path_to_output, output_motl_name))
print(full_names)
