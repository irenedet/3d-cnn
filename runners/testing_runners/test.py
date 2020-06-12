# import numpy as np
# import os
#
# import h5py
# import numpy as np
# from tqdm import tqdm

# input_paths = [
#     "/struct/mahamid/Irene/yeast/healthy/180426/030/grid_partition.h5",
#     "/struct/mahamid/Irene/yeast/healthy/180426/034/grid_partition.h5",
#     "/struct/mahamid/Irene/yeast/healthy/180426/037/grid_partition.h5",
#     "/struct/mahamid/Irene/yeast/healthy/180426/043/grid_partition.h5",
#     "/struct/mahamid/Irene/yeast/healthy/180426/045/grid_partition.h5",
#     ]
#
# output_paths = [
#     "/struct/mahamid/Irene/yeast/healthy/180426/030/strongly_labeled_min0.01_max1/single_filter_64pix/full_partition.h5",
#     "/struct/mahamid/Irene/yeast/healthy/180426/034/strongly_labeled_min0.01_max1/single_filter_64pix/full_partition.h5",
#     "/struct/mahamid/Irene/yeast/healthy/180426/037/strongly_labeled_min0.01_max1/single_filter_64pix/full_partition.h5",
#     "/struct/mahamid/Irene/yeast/healthy/180426/043/strongly_labeled_min0.01_max1/single_filter_64pix/full_partition.h5",
#     "/struct/mahamid/Irene/yeast/healthy/180426/045/strongly_labeled_min0.01_max1/single_filter_64pix/full_partition.h5",
# ]
#
# label = 'FAS_non_sph_masks_64pix_encoder_dropout0_decoder_dropout0.2_DA_DA_G5_E0_R180_SP0.04_DArounds6_BN_False_fas__D_2_IF_32_set_1'
# label_path = os.path.join('volumes/predictions', label)
#
# for input_path, output_path in zip(input_paths, output_paths):
#     with h5py.File(input_path, "r") as f:
#         with h5py.File(output_path, "a") as o:
#             labeled_subtomos = list(f[label_path])
#             o_raw_subtomos = list(o['volumes/raw'])
#             n = len(o_raw_subtomos)
#             for index, subtomo_name in zip(tqdm(range(n)),o_raw_subtomos):
#                 if subtomo_name in labeled_subtomos:
#                     subtomo_path = os.path.join(label_path, subtomo_name)
#                     subtomo_data = f[subtomo_path][:]
#                     o[subtomo_path] = subtomo_data

# motl_paths = [
#     # "/struct/mahamid/Irene/yeast/ED/190301/001/clean_motls/ribo/combined/TM_cnnIF4_cnnIF8_cnnIF32_motl_1411.csv",
#     # "/struct/mahamid/Irene/yeast/ED/190301/002/clean_motls/ribo/combined/TM_cnnIF4_cnnIF8_cnnIF32_motl_1311.csv",
#     # "/struct/mahamid/Irene/yeast/ED/190301/006/clean_motls/ribo/combined/TM_cnnIF4_cnnIF8_cnnIF32_motl_1550.csv",
#     # "/struct/mahamid/Irene/yeast/ED/190301/010/clean_motls/ribo/combined/TM_cnnIF4_cnnIF8_cnnIF32_motl_1265.csv",
#     # "/struct/mahamid/Irene/yeast/ED/190301/011/clean_motls/ribo/combined/TM_cnnIF4_cnnIF8_cnnIF32_motl_1238.csv",
#     # "/struct/mahamid/Irene/yeast/ED/190301/013/clean_motls/ribo/combined/TM_cnnIF4_cnnIF8_cnnIF32_motl_1274.csv",
#     # "/struct/mahamid/Irene/yeast/ED/190301/015/clean_motls/ribo/combined/TM_cnnIF4_cnnIF8_cnnIF32_motl_1259.csv",
#     # "/struct/mahamid/Irene/yeast/ED/190301/020/clean_motls/ribo/combined/TM_cnnIF4_cnnIF8_cnnIF32_motl_1663.csv",
#     # "/struct/mahamid/Irene/yeast/ED/190301/021/clean_motls/ribo/combined/TM_cnnIF4_cnnIF8_cnnIF32_motl_1091.csv",
#     # "/struct/mahamid/Irene/yeast/ED/190301/026/clean_motls/ribo/combined/TM_cnnIF4_cnnIF8_cnnIF32_motl_1310.csv",
#     # "/struct/mahamid/Irene/yeast/ED/190301/029/clean_motls/ribo/combined/TM_cnnIF4_cnnIF8_cnnIF32_motl_1694.csv",
#     # "/struct/mahamid/Irene/yeast/ED/190301/030/clean_motls/ribo/combined/TM_cnnIF4_cnnIF8_cnnIF32_motl_1400.csv",
#     # "/struct/mahamid/Irene/yeast/ED/190301/004/clean_motls/ribo/combined/TM_cnnIF4_cnnIF8_cnnIF32_motl_1256.csv",
#     # "/struct/mahamid/Irene/yeast/ED/190301/007/clean_motls/ribo/combined/TM_cnnIF4_cnnIF8_cnnIF32_motl_822.csv",
#     # "/struct/mahamid/Irene/yeast/ED/190301/014/clean_motls/ribo/combined/TM_cnnIF4_cnnIF8_cnnIF32_motl_742.csv",
#     # "/struct/mahamid/Irene/yeast/ED/190301/017/clean_motls/ribo/combined/TM_cnnIF4_cnnIF8_cnnIF32_motl_1082.csv",
#     # "/struct/mahamid/Irene/yeast/ED/190301/018/clean_motls/ribo/combined/TM_cnnIF4_cnnIF8_cnnIF32_motl_1098.csv",
#     # "/struct/mahamid/Irene/yeast/ED/190301/019/clean_motls/ribo/combined/TM_cnnIF4_cnnIF8_cnnIF32_motl_1105.csv",
# ]
#
# mask_paths = [
#     # "/struct/mahamid/Irene/yeast/ED/190301/001/clean_masks/TM_cnnIF4_cnnIF8_cnnIF32_motl_1411.mrc",
#     # "/struct/mahamid/Irene/yeast/ED/190301/002/clean_masks/TM_cnnIF4_cnnIF8_cnnIF32_motl_1311.mrc",
#     # "/struct/mahamid/Irene/yeast/ED/190301/006/clean_masks/TM_cnnIF4_cnnIF8_cnnIF32_motl_1550.mrc",
#     # "/struct/mahamid/Irene/yeast/ED/190301/010/clean_masks/TM_cnnIF4_cnnIF8_cnnIF32_motl_1265.mrc",
#     # "/struct/mahamid/Irene/yeast/ED/190301/011/clean_masks/TM_cnnIF4_cnnIF8_cnnIF32_motl_1238.mrc",
#     # "/struct/mahamid/Irene/yeast/ED/190301/013/clean_masks/TM_cnnIF4_cnnIF8_cnnIF32_motl_1274.mrc",
#     # "/struct/mahamid/Irene/yeast/ED/190301/015/clean_masks/TM_cnnIF4_cnnIF8_cnnIF32_motl_1259.mrc",
#     # "/struct/mahamid/Irene/yeast/ED/190301/020/clean_masks/TM_cnnIF4_cnnIF8_cnnIF32_motl_1663.mrc",
#     # "/struct/mahamid/Irene/yeast/ED/190301/021/clean_masks/TM_cnnIF4_cnnIF8_cnnIF32_motl_1091.mrc",
#     # "/struct/mahamid/Irene/yeast/ED/190301/026/clean_masks/TM_cnnIF4_cnnIF8_cnnIF32_motl_1310.mrc",
#     # "/struct/mahamid/Irene/yeast/ED/190301/029/clean_masks/TM_cnnIF4_cnnIF8_cnnIF32_motl_1694.mrc",
#     # "/struct/mahamid/Irene/yeast/ED/190301/030/clean_masks/TM_cnnIF4_cnnIF8_cnnIF32_motl_1400.mrc",
#     # "/struct/mahamid/Irene/yeast/ED/190301/004/clean_masks/TM_cnnIF4_cnnIF8_cnnIF32_motl_1256.mrc",
#     # "/struct/mahamid/Irene/yeast/ED/190301/007/clean_masks/TM_cnnIF4_cnnIF8_cnnIF32_motl_822.mrc",
#     # "/struct/mahamid/Irene/yeast/ED/190301/014/clean_masks/TM_cnnIF4_cnnIF8_cnnIF32_motl_742.mrc",
#     # "/struct/mahamid/Irene/yeast/ED/190301/017/clean_masks/TM_cnnIF4_cnnIF8_cnnIF32_motl_1082.mrc",
#     # "/struct/mahamid/Irene/yeast/ED/190301/018/clean_masks/TM_cnnIF4_cnnIF8_cnnIF32_motl_1098.mrc",
#     # "/struct/mahamid/Irene/yeast/ED/190301/019/clean_masks/TM_cnnIF4_cnnIF8_cnnIF32_motl_1105.mrc",
# ]
#
# # output_shape = (500, 928, 928)
# # radius = 8
# # values_in_motl = False
# # z_shift = 0
# # coords_in_tom_format = True
# #
# # for motl_path, output_path in zip(motl_paths, mask_paths):
# #     print(motl_path)
# #
# #     write_particle_mask_from_motl(path_to_motl=motl_path,
# #                                   output_path=output_path,
# #                                   output_shape=output_shape,
# #                                   sphere_radius=radius,
# #                                   values_in_motl=values_in_motl,
# #                                   number_of_particles=None,
# #                                   z_shift=z_shift,
# #                                   particles_in_tom_format=coords_in_tom_format)
# #
# #
#
#
# import numpy as np
#
# from file_actions.readers.tomograms import load_tomogram
# from file_actions.writers.mrc import write_mrc_dataset
#
# paths = [
#     "/struct/mahamid/Irene/yeast/ED/181119/002/alex_labels.hdf",
#     "/struct/mahamid/Irene/yeast/ED/181119/030/alex_labels.hdf",
#     "/struct/mahamid/Irene/yeast/ED/190329/001/alex_labels.hdf",
#     "/struct/mahamid/Irene/yeast/ED/190329/004/alex_labels.hdf",
#     # "/struct/mahamid/Irene/yeast/healthy/180426/021/alex_labels.hdf",
#     # "/struct/mahamid/Irene/yeast/healthy/180426/024/alex_labels.hdf",
#     # "/struct/mahamid/Irene/yeast/healthy/180426/026/alex_labels.hdf",
#     # "/struct/mahamid/Irene/yeast/healthy/180426/027/alex_labels.hdf",
#     # "/struct/mahamid/Irene/yeast/healthy/180711/003/alex_labels.hdf",
#     # "/struct/mahamid/Irene/yeast/healthy/180711/004/alex_labels.hdf",
# ]
#
# flips = [
#     False,  # /struct/mahamid/Irene/yeast/ED/181119/002/alex_labels.hdf
#     False,  # /struct/mahamid/Irene/yeast/ED/181119/030/alex_labels.hdf
#     True,  # /struct/mahamid/Irene/yeast/ED/190329/001/alex_labels.hdf
#     True,  # /struct/mahamid/Irene/yeast/ED/190329/004/alex_labels.hdf
#     # True,  # /struct/mahamid/Irene/yeast/healthy/180426/021/alex_labels.hdf
#     # True,  # /struct/mahamid/Irene/yeast/healthy/180426/024/alex_labels.hdf
#     # False,  # /struct/mahamid/Irene/yeast/healthy/180426/026/alex_labels.hdf
#     # False,  # /struct/mahamid/Irene/yeast/healthy/180426/027/alex_labels.hdf
#     # True,  # /struct/mahamid/Irene/yeast/healthy/180711/003/alex_labels.hdf
#     # True,  # /struct/mahamid/Irene/yeast/healthy/180711/004/alex_labels.hdf
# ]
# # path = "/struct/mahamid/Irene/yeast/healthy/180426/005/alex_labels.hdf"
# assert len(flips) == len(paths)
# for path, flip in zip(paths, flips):
#     tomo = load_tomogram(path_to_dataset=path)
#     tomo_name = path[-26:-16]
#     print(tomo_name)
#     id = 1
#     tomo = 1 * (tomo == 1)
#     output_path = "/struct/mahamid/Irene/yeast/ED/"+ tomo_name +"/cytosol_mask.mrc"
#     if flip:
#         tomo = np.flip(tomo, axis=1)
#
#     write_mrc_dataset(mrc_path=output_path, array=tomo, dtype=np.int8)

import numpy as np
import pandas as pd

motl_path = "/Users/trueba/struct/mahamid/Irene/yeast/healthy/180426/004/ribos/motl/corrected_motl_verified.csv"
output_mask_path = "/Users/trueba/struct/mahamid/Irene/3d-cnn/runners/quantifications/test_data/cube_mask.mrc"
output_cube_motl = "/Users/trueba/struct/mahamid/Irene/3d-cnn/runners/quantifications/test_data/cube_motl.csv"

length = 100
lower_corner = np.array([500, 570, 500])
upper_corner = lower_corner + length * np.ones((1, 3))

mask = np.ones((length, length, length))
from tomogram_utils.peak_toolbox.utils import read_motl_coordinates_and_values

# from file_actions.writers.datasets import write_dataset
# write_dataset(output_path=output_mask_path, array=mask)
_, motl_coords = read_motl_coordinates_and_values(path_to_motl=motl_path)
output_cube_motl_df = pd.DataFrame({})

coord_in_cube = []
for p in motl_coords:
    x, y, z = p
    numpy_p = np.array(p)
    if np.sum(numpy_p >= lower_corner) == 3:
        if np.sum(numpy_p < upper_corner) == 3:
            shifted_p = numpy_p - lower_corner
            coord_in_cube.append(shifted_p)
print("Coordinates in cube: {}".format(len(coord_in_cube)))

df = pd.DataFrame({})
df["x"] = [p[0] for p in coord_in_cube]
df["y"] = [p[1] for p in coord_in_cube]
df["z"] = [p[2] for p in coord_in_cube]
print(df)
df.to_csv(output_cube_motl, index=False)
