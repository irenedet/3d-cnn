# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from performance.statistics_utils import pr_auc_score, \
#     f1_score_calculator, precision_recall_calculator
# from tomogram_utils.peak_toolbox.utils import read_motl_coordinates_and_values
# import shutil
# import os
#
# from constants.dataset_tables import DatasetTableHeader
# from file_actions.readers.tomograms import load_tomogram
# from tomogram_utils.peak_toolbox.utils import read_motl_coordinates_and_values
#
# # from file_actions.writers.mrc import write_mrc_dataset
#
#
# import csv
# from os.path import join
#
#
# def motl_writer(path_to_output_folder: str, list_of_peak_scores: list,
#                 list_of_peak_coords: list, in_tom_format=False,
#                 order_by_score=True, list_of_angles: list or bool = False,
#                 motl_name: None or str = None):
#     """
#     Already modified to match em_motl format
#     Format of MOTL:
#        The following parameters are stored in the matrix MOTIVELIST of dimension
#        (20, NPARTICLES)<= ToDo This! (currently storing (NPARTICLES, 20)):
#        column
#           1         : Score Coefficient from localisation algorithm
#           2         : x-coordinate in full tomogram
#           3         : y-coordinate in full tomogram
#           4         : peak number
#           5         : running index of tilt series (optional)
#           8         : x-coordinate in full tomogram
#           9         : y-coordinate in full tomogram
#           10        : z-coordinate in full tomogram
#           14        : x-shift in subvolume (AFTER rotation of reference)
#           15        : y-shift in subvolume
#           16        : z-shift in subvolume
#           17        : Phi
#           18        : Psi
#           19        : Theta
#           20        : class number
#     For more information check tom package documentation (e.g. tom_chooser.m).
#     """
#     numb_peaks = len(list_of_peak_scores)
#     joint_list = list(zip(list_of_peak_scores, list_of_peak_coords))
#
#     if order_by_score:
#         print("saving coordinates ordered by decreasing score value")
#         joint_list = sorted(joint_list, key=lambda pair: pair[0], reverse=1)
#     else:
#         print("saving coordinates without sorting by score value")
#
#     if motl_name is None:
#         motl_file_name = join(path_to_output_folder,
#                               'motl_' + str(numb_peaks) + '.csv')
#     else:
#         motl_file_name = join(path_to_output_folder, motl_name)
#     with open(motl_file_name, 'w', newline='') as csvfile:
#         motlwriter = csv.writer(csvfile, delimiter=' ', quotechar='|',
#                                 quoting=csv.QUOTE_MINIMAL)
#         for index, tuple_val_point in enumerate(joint_list):
#             val, point = tuple_val_point
#             if in_tom_format:
#                 x, y, z = point
#             # else:
#             #     x, y, z = to_tom_coordinate_system(point)
#             coordinate_in_tom_format = str(x) + ',' + str(y) + ',' + str(z)
#             if not list_of_angles:
#                 angle_str = '0,0,0'
#             else:
#                 phi, psi, theta = list_of_angles[index]
#                 angle_str = str(phi) + ',' + str(psi) + ',' + str(theta)
#             xy_columns = ',' + str(x) + ',' + str(y) + ','
#             class_str = '1'  # by default, maybe useful to list in arguments
#             tail = ',0,0,0,0,0,0,' + angle_str + ',' + class_str
#
#             row = str(val) + xy_columns + str(
#                 index) + ',0,0,0,' + coordinate_in_tom_format + tail
#             motlwriter.writerow([row])
#     print("The motive list has been writen in", motl_file_name)
#     return motl_file_name
#
#
# # motl_path = "/Users/trueba/mnt/struct/mahamid/Irene/yeast/healthy/180426/004/fas/motl/corrected_motl_191108.csv"
# # output_path = "/Users/trueba/mnt/struct/mahamid/Irene/yeast/healthy/180426/004/clean_masks/fas/"
# #
# # motl_vals, motl_coords = read_motl_coordinates_and_values(path_to_motl=motl_path)
# # motl_coords = [[p[0] + 16, p[1], p[2]] for p in motl_coords]
# #
# # motl_writer(path_to_output_folder=output_path,
# #             list_of_peak_scores=motl_vals,
# #             list_of_peak_coords=motl_coords,
# #             in_tom_format=True,
# #             motl_name="corrected_motl_191108_shifted_by_16.csv")
#
#
# tomo_names = [
#     "ScED_6h/001",
#     "ScED_6h/002",
#     "ScED_6h/003",
#     "ScED_6h/006",
#     "ScED_6h/011"
# ]
#
# motl_paths = [
#     "/struct/mahamid/Irene/scratch/trueba/3d-cnn/cnn_evaluation/ribo_sph_masks_TM_CNNs/64pix_encoder_dropout0_decoder_dropout0_DA_none_BN_False_ribo__D_2_IF_4_set_1/ScED_6h/001/ribo/motl_1963.csv",
#     "/struct/mahamid/Irene/scratch/trueba/3d-cnn/cnn_evaluation/ribo_sph_masks_TM_CNNs/64pix_encoder_dropout0_decoder_dropout0_DA_none_BN_False_ribo__D_2_IF_4_set_1/ScED_6h/002/ribo/motl_919.csv",
#     "/struct/mahamid/Irene/scratch/trueba/3d-cnn/cnn_evaluation/ribo_sph_masks_TM_CNNs/64pix_encoder_dropout0_decoder_dropout0_DA_none_BN_False_ribo__D_2_IF_4_set_1/ScED_6h/003/ribo/motl_2147.csv",
#     "/struct/mahamid/Irene/scratch/trueba/3d-cnn/cnn_evaluation/ribo_sph_masks_TM_CNNs/64pix_encoder_dropout0_decoder_dropout0_DA_none_BN_False_ribo__D_2_IF_4_set_1/ScED_6h/006/ribo/motl_597.csv",
#     "/struct/mahamid/Irene/scratch/trueba/3d-cnn/cnn_evaluation/ribo_sph_masks_TM_CNNs/64pix_encoder_dropout0_decoder_dropout0_DA_none_BN_False_ribo__D_2_IF_4_set_1/ScED_6h/011/ribo/motl_1765.csv",
#     # "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/warp/ScED_6h/tilts/imod/001/TM/001_motl_ribo_3000.em",
#     # "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/warp/ScED_6h/tilts/imod/002/TM/002_motl_ribo_3000.em",
#     # "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/warp/ScED_6h/tilts/imod/003/TM/003_motl_ribo_3000.em",
#     # "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/warp/ScED_6h/tilts/imod/006/TM/006_motl_ribo_3000.em",
#     # "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/warp/ScED_6h/tilts/imod/011/TM/011_motl_ribo_3000.em",
# ]
# output_dir = "/struct/mahamid/Irene/scratch/trueba/3d-cnn/cnn_evaluation/ribo_sph_masks_TM_CNNs/64pix_encoder_dropout0_decoder_dropout0_DA_none_BN_False_ribo__D_2_IF_4_set_1"
# precision_global = []
# recall_global = []
# legend_str_global = []
#
# for tomo_name, motl_path in zip(tomo_names, motl_paths):
#     output_path = os.path.join(output_dir, tomo_name)
#     output_path = os.path.join(output_path, "ribo/")
#     figures_dir = os.path.join(output_path, "figures")
#     os.makedirs(figures_dir, exist_ok=True)
#     dataset_table = "/struct/mahamid/Irene/yeast/yeast_table.csv"

# mask = "cytosol"
# particle = "ribo"
# sample_type = "healthy"
# mask_value = 1
#
# df = pd.read_csv(dataset_table)
# DTHeader_masks = DatasetTableHeader(semantic_classes=[mask])
# DTHeader_particles = DatasetTableHeader(semantic_classes=[particle])
#
# tomo_row = df.loc[df[DTHeader_masks.tomo_name] == tomo_name]
# mask_column = DTHeader_masks.masks_names[0]
# mask_path = tomo_row[mask_column].values[0]
# print(mask_path)
# mask_array = load_tomogram(mask_path)
# mask_array_coords = list(np.transpose(np.where(mask_array == 1)))
# print("number of vals in mask", len(mask_array_coords))
# mask_array_coords = [tuple(point) for point in mask_array_coords]
#
# assert len(mask_array_coords) > 0, "The domain mask is empty."
#
# motl_vals, motl_coords = read_motl_coordinates_and_values(path_to_motl=motl_path)
# coords = [(p[2], p[1], p[0]) for p in motl_coords]
# before_filtering_n = len(coords)
# coords = set(coords) & set(mask_array_coords)
# coords = [list(p) for p in coords]
# predicted_values = motl_vals
# predicted_coordinates = [[p[2], p[1], p[0]] for p in coords]
# print("{}: before filtering: n = {}, after filtering: n = {}".format(tomo_name, before_filtering_n, len(coords)))

#     clean_motl = "path_to_motl_clean_ribo"
#     DTHeader = DatasetTableHeader(clean_motl=clean_motl)
#     df = pd.read_csv(dataset_table)
#     df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)
#     tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
#     radius = 10  # config['performance_parameters']['pr_tolerance_radius']
#     box_shape = 64  # config['box_shape']
#     overlap = 12
#     if isinstance(box_shape, int):
#         box_shape = [box_shape, box_shape, box_shape]
#     else:
#         box_shape = tuple([int(elem) for elem in reversed(box_shape)])
#
#     assert len(box_shape) == 3, "Invalid box_shape"
#
#     z_shift = int(tomo_df.iloc[0][DTHeader.z_shift])
#     y_shift = int(tomo_df.iloc[0][DTHeader.y_shift])
#     x_shift = int(tomo_df.iloc[0][DTHeader.x_shift])
#     x_dim = int(tomo_df.iloc[0][DTHeader.x_dim])
#     y_dim = int(tomo_df.iloc[0][DTHeader.y_dim])
#     z_dim = int(tomo_df.iloc[0][DTHeader.z_dim])
#
#     path_to_motl_true = tomo_df.iloc[0][DTHeader.clean_motls]
#
#     print("")
#     print("path_to_motl_true = ", path_to_motl_true)
#     print("output_path = ", output_path)
#     dataset_shape = (z_dim, y_dim, x_dim)
#     subtomo_shape = tuple([sh - overlap for sh in box_shape])
#
#     class_number = 0  # config['reconstruction']['class_number']
#     class_name = ["ribo"]  # semantic_classes[class_number]
#     predicted_values, predicted_coordinates = read_motl_coordinates_and_values(path_to_motl=motl_path)
#     true_values, true_coordinates = read_motl_coordinates_and_values(
#         path_to_motl=path_to_motl_true)
#     unique_peaks_number = len(predicted_values)
#
#     shift_vector = np.array([x_shift, y_shift, 0])
#     predicted_coordinates = [np.array(p) + shift_vector for p in
#                              predicted_coordinates]
#     predicted_coordinates = np.array(predicted_coordinates)
#
#     prec, recall, tp_true, tp_pred, fp_pred, tp_pred_scores, fp_pred_scores, *_ = \
#         precision_recall_calculator(
#             predicted_coordinates=predicted_coordinates,
#             value_predicted=predicted_values,
#             true_coordinates=true_coordinates,
#             radius=radius)
#
#     F1_score = f1_score_calculator(prec, recall)
#
#     if len(F1_score) > 0:
#         max_F1 = np.max(F1_score)
#         optimal_peak_number = np.min(np.where(F1_score == max_F1)[0])
#     else:
#         max_F1 = 0
#         optimal_peak_number = np.nan
#
#     auPRC = pr_auc_score(precision=prec, recall=recall)
#     print("auPRC = ", auPRC, "and max_F1 = ", max_F1)
#
#     output_path = os.path.join(output_path, "pr_radius_" + str(radius))
#     path_to_detected_predicted = join(output_path, "detected")
#     path_to_undetected_predicted = join(output_path, "undetected")
#
#     os.makedirs(path_to_detected_predicted, exist_ok=True)
#     os.makedirs(path_to_undetected_predicted, exist_ok=True)
#
#     motl_writer(path_to_output_folder=path_to_detected_predicted,
#                 list_of_peak_coords=tp_pred,
#                 list_of_peak_scores=tp_pred_scores,
#                 in_tom_format=True)
#     motl_writer(path_to_output_folder=path_to_undetected_predicted,
#                 list_of_peak_coords=fp_pred,
#                 list_of_peak_scores=fp_pred_scores,
#                 in_tom_format=True)
#
#     # plt.figure(1)
#     # plt.hist(predicted_values, bins=45, label="predicted")
#     # plt.xlabel("score value")
#     # plt.ylabel("frequency")
#     # plt.title(str(len(predicted_values)) + " peaks")
#     # plt.legend()
#     # plt.gcf()
#     # fig_name = join(figures_dir, "histogram_predicted.png")
#     # plt.savefig(fname=fig_name, format="png")
#     #
#     # plt.figure(2)
#     # plt.hist(tp_pred_scores, bins=45, label="true positives",
#     #          fc=(0, 0, 1, 0.5))
#     # plt.hist(fp_pred_scores, bins=45, label="false positives",
#     #          fc=(1, 0, 0, 0.5))
#     # plt.xlabel("score value")
#     # plt.ylabel("frequency")
#     # plt.title(str(len(fp_pred)) + " peaks")
#     # plt.legend()
#     # plt.gcf()
#     # fig_name = join(figures_dir, "histogram-detected-undetected.png")
#     # plt.savefig(fname=fig_name, format="png")
#     #
#     # plt.figure(3)
#     # pr_legend_str = "auPRC = " + str(round(auPRC, 4))
#     # f1_legend_str = "(max_F1, best_peaks) = (" + str(
#     #     round(max_F1, 4)) + ", " + str(optimal_peak_number) + ")"
#     # title_str = str(len(predicted_coordinates)) + " peaks"
#     # plt.plot(F1_score, label=f1_legend_str)
#     # plt.xlabel("number of peaks")
#     # plt.ylabel("F1 score")
#     # plt.xlim((0, 1))
#     # plt.ylim((0, 1))
#     # plt.title(title_str)
#     # plt.legend()
#     # plt.gcf()
#     # fig_name = join(figures_dir, "f1_score_" + title_str + ".png")
#     # plt.savefig(fname=fig_name, format="png")
#     # plt.close(3)
#
#     # plt.figure(4)
#     # pr_legend_str = "auPRC={}".format(round(auPRC, 4))
#     # plt.plot(recall, prec, label=pr_legend_str)
#     # plt.xlabel("recall")
#     # plt.ylabel("precision")
#     # plt.legend()
#     # plt.xlim((0, 1))
#     # plt.ylim((0, 1))
#     # plt.gcf()
#     # title_str = str(len(predicted_coordinates))
#     # plt.title(title_str)
#     # fig_name = join(figures_dir, "pr_" + title_str + ".png")
#     # plt.savefig(fname=fig_name, format="png")
#     # plt.close()
#     precision_global.append(prec)
#     recall_global.append(recall)
#     legend_str_global.append("{}: auPRC={}".format(tomo_name, round(auPRC, 2)))
#
# plt.figure(5)
# for prec, recall, pr_legend_str in zip(precision_global, recall_global, legend_str_global):
#     # pr_legend_str = "{}: auPRC={}".format(tomo_name, round(auPRC, 2))
#     plt.plot(recall, prec, label=pr_legend_str)
#     plt.xlabel("recall")
#     plt.ylabel("precision")
#     plt.legend()
#     plt.xlim((0, 1))
#     plt.ylim((0, 1))
#     plt.gcf()
# title_str = "combined"
# plt.title(title_str)
# fig_name = join("/struct/mahamid/Irene/yeast/ScED_6h/", "TM_vs_CNN_pr_combined_NO_cytosol_filtering.png")
# plt.savefig(fname=fig_name, format="png")
#     # if statistics_file is not None:
#     #     statistics_label = segmentation_label + "_pr_radius_" + str(radius)
#     #
#     #     write_statistics(statistics_file=statistics_file,
#     #                      statistics_label="auPRC_" + statistics_label,
#     #                      tomo_name=tomo_name,
#     #                      stat_measure=round(auPRC, 4))
#     #
#     #     write_statistics(statistics_file=statistics_file,
#     #                      statistics_label="F1_" + statistics_label,
#     #                      tomo_name=tomo_name,
#     #                      stat_measure=round(max_F1, 4))


# with (open(filepath, "rb")) as openfile:
#     while True:
#         try:
#             objects.append(pickle.load(openfile))
#         except EOFError:
#             break

# import matplotlib.pyplot as plt
# import mrcfile
# import numpy as np
# from file_actions.readers.mrc import read_mrc
# from file_actions.readers.motl import read_motl_from_csv
# import pandas as pd
# import os
#
# global_motl_path = "/Users/trueba/mnt/struct/mahamid/Irene/yeast/healthy/fas_on_motl.csv"
# global_motl = pd.read_csv(filepath_or_buffer=global_motl_path)
#
# tomos_motl_df = [(x, pd.DataFrame(y)) for x, y in global_motl.groupby('tomo_name')]
# for tomo_name, tomo_data in tomos_motl_df:
#     print(tomo_name)
#     output_path = os.path.join("/Users/trueba/mnt/struct/mahamid/Irene/yeast/healthy/", tomo_name)
#     output_path = os.path.join(output_path, "fas/SpYES_motl_cnn_vs_TM.csv")
#     dirname = os.path.dirname(output_path)
#     # print(dirname)
#     os.makedirs(dirname, exist_ok=True)
#     subtomo_data = tomo_data.loc[:, tomo_data.columns != 'tomo_name']
#     # print(subtomo_data)
#     subtomo_data[['x', 'y', 'z']] = subtomo_data[['x', 'y', 'z']]*0.25
#     # print(subtomo_data)
#     subtomo_data.to_csv(output_path, header=False, index=False)

from file_actions.readers.tomograms import load_tomogram
from file_actions.writers.mrc import write_mrc_dataset
actine_mask = load_tomogram(path_to_dataset="/struct/mahamid/Irene/mauricio/actin_masks/induced_mask_nonbinary.mrc")

actine_mask = 1 * (actine_mask > 0)
write_mrc_dataset(mrc_path="/struct/mahamid/Irene/mauricio/actin_masks/Induced_actin.mrc", array=actine_mask)
