# from file_actions.readers.mrc import read_mrc
# from file_actions.readers.tomograms import load_tomogram
#
# data_path = "/struct/mahamid/Mauro/DATA_TITAN/Tomo_Processing/20200107_Sergio_Induced/Induced/Tilt_series/Sorted/Fiducials/00010/00010_sq_df_sorted.rec"
#
# # tomo = read_mrc(path_to_mrc=data_path)
# #
# # print(tomo.dtype)
# #
# # tomo = tomo.astype(float)
# #
# # print(tomo.dtype)
#
# tomo = load_tomogram(path_to_dataset=data_path, dtype=float)
# print(tomo.dtype)


ori_masks = [
    # "/struct/mahamid/Irene/yeast/healthy/180426/026/clean_masks/ribo/TM_mask/ribo_non_sph_mas_thr.mrc",
    "/struct/mahamid/Irene/yeast/healthy/180426/027/clean_masks/ribo/TM_mask/ribo_non_sph_mas_thr.mrc",
    "/struct/mahamid/Irene/yeast/healthy/180426/028/clean_masks/ribo/TM_mask/ribo_non_sph_mas_thr.mrc",
    "/struct/mahamid/Irene/yeast/healthy/180426/029/clean_masks/ribo/TM_mask/ribo_non_sph_mas_thr.mrc",
    "/struct/mahamid/Irene/yeast/healthy/180426/030/clean_masks/ribo/TM_mask/ribo_non_sph_mas_thr.mrc",
    "/struct/mahamid/Irene/yeast/healthy/180426/034/clean_masks/ribo/TM_mask/ribo_non_sph_mas_thr.mrc",
    "/struct/mahamid/Irene/yeast/healthy/180426/037/clean_masks/ribo/TM_mask/ribo_non_sph_mas_thr.mrc",
    "/struct/mahamid/Irene/yeast/healthy/180426/041/clean_masks/ribo/TM_mask/ribo_non_sph_mas_thr.mrc",
    "/struct/mahamid/Irene/yeast/healthy/180426/043/clean_masks/ribo/TM_mask/ribo_non_sph_mas_thr.mrc",
    "/struct/mahamid/Irene/yeast/healthy/180426/045/clean_masks/ribo/TM_mask/ribo_non_sph_mas_thr.mrc",
]

extra_masks = [
    # "/struct/mahamid/Irene/yeast/healthy/180426/026/prediction_cnn_IF4_IF8_IF32/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_351.mrc",
    "/struct/mahamid/Irene/yeast/healthy/180426/027/prediction_cnn_IF4_IF8_IF32/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_358.mrc",
    "/struct/mahamid/Irene/yeast/healthy/180426/028/prediction_cnn_IF4_IF8_IF32/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_598.mrc",
    "/struct/mahamid/Irene/yeast/healthy/180426/029/prediction_cnn_IF4_IF8_IF32/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_786.mrc",
    "/struct/mahamid/Irene/yeast/healthy/180426/030/prediction_cnn_IF4_IF8_IF32/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_615.mrc",
    "/struct/mahamid/Irene/yeast/healthy/180426/034/prediction_cnn_IF4_IF8_IF32/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_310.mrc",
    "/struct/mahamid/Irene/yeast/healthy/180426/037/prediction_cnn_IF4_IF8_IF32/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_352.mrc",
    "/struct/mahamid/Irene/yeast/healthy/180426/041/prediction_cnn_IF4_IF8_IF32/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_1249.mrc",
    "/struct/mahamid/Irene/yeast/healthy/180426/043/prediction_cnn_IF4_IF8_IF32/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_499.mrc",
    "/struct/mahamid/Irene/yeast/healthy/180426/045/prediction_cnn_IF4_IF8_IF32/ribo/in_cytosol_mask/pr_radius_10/undetected/motl_514.mrc",
]
import os

from file_actions.readers.tomograms import load_tomogram
from file_actions.writers.mrc import write_mrc_dataset

for file, extra_file in zip(ori_masks, extra_masks):
    ori_mask = load_tomogram(path_to_dataset=file)
    extra_mask = load_tomogram(path_to_dataset=extra_file)

    shz, shy, shx = ori_mask.shape

    new_mask = extra_mask.copy()
    new_mask[:shz, :shy, :shx] += ori_mask

    new_mask = 1 * (new_mask > 0)
    #
    basedir_output = os.path.dirname(file)[:-7]
    print(basedir_output)
    mrc_path = os.path.join(basedir_output, "TM_added_IF4_IF8_IF32_undetected/ribo_mask.mrc")
    os.makedirs(os.path.dirname(mrc_path))

    write_mrc_dataset(mrc_path=mrc_path, array=extra_mask)
