import os

import numpy as np

from file_actions.readers.tomograms import load_tomogram
from file_actions.writers.mrc import write_mrc_dataset

tomos = [
    "/struct/mahamid/Irene/shrec2020/data/shrec2020",
    # "/struct/mahamid/Irene/shrec2020/data/shrec2020/model_0",
    # "/struct/mahamid/Irene/shrec2020/data/shrec2020/model_1",
    # "/struct/mahamid/Irene/shrec2020/data/shrec2020/model_2",
    # "/struct/mahamid/Irene/shrec2020/data/shrec2020/model_3",
    # "/struct/mahamid/Irene/shrec2020/data/shrec2020/model_4",
    # "/struct/mahamid/Irene/shrec2020/data/shrec2020/model_5",
    # "/struct/mahamid/Irene/shrec2020/data/shrec2020/model_6",
    # "/struct/mahamid/Irene/shrec2020/data/shrec2020/model_7",
    # "/struct/mahamid/Irene/shrec2020/data/shrec2020/model_8",
]

# particles = list(range(1, 13))
output_shape = (512, 512, 512)
# threshold = 0.2
value = 1


def get_file_basename(file):
    base = os.path.basename(file)
    basename, ext = os.path.splitext(base)
    return basename


for input_dir in tomos:
    particle_name = "lamella.mrc"
    # input_path = os.path.join(input_dir, "class_mask_sq.mrc")
    output_path = os.path.join(input_dir, particle_name)
    print(output_path)
    # input_data = load_tomogram(path_to_dataset=input_path)
    output_data = np.zeros(output_shape)
    middle_slice = slice(156, 356)
    output_data[middle_slice, :, :] = np.ones((200, 512, 512))
    # output_data = value * (input_data > 0)
    write_mrc_dataset(mrc_path=output_path, array=output_data,
                      dtype=np.int8)
print("finished!")

# path_to_motls = [
#     "/struct/mahamid/Irene/yeast/healthy/180426/004/2bin/fas/motl/corrected_motl_191108.csv",
#     # "/struct/mahamid/Irene/yeast/healthy/180426/005/2bin/fas/motl/corrected_motl_191108.csv",
#     # "/struct/mahamid/Irene/yeast/healthy/180426/021/2bin/fas/motl/corrected_motl_191108.csv",
#     # "/struct/mahamid/Irene/yeast/healthy/180426/024/2bin/fas/motl/corrected_motl_191108.csv",
#     # "/struct/mahamid/Irene/yeast/healthy/180426/026/2bin/fas/motl/motl_job003.csv",
#     # "/struct/mahamid/Irene/yeast/healthy/180426/027/2bin/fas/motl/motl_job003.csv",
#     # "/struct/mahamid/Irene/yeast/healthy/180426/028/2bin/fas/motl/motl_job003.csv",
#     # "/struct/mahamid/Irene/yeast/healthy/180426/029/2bin/fas/motl/motl_job003.csv",
#     # "/struct/mahamid/Irene/yeast/healthy/180426/030/2bin/fas/motl/motl_job003.csv",
#     # "/struct/mahamid/Irene/yeast/healthy/180426/034/2bin/fas/motl/motl_job003.csv",
#     # "/struct/mahamid/Irene/yeast/healthy/180426/037/2bin/fas/motl/motl_job003.csv",
#     # "/struct/mahamid/Irene/yeast/healthy/180426/041/2bin/fas/motl/motl_job003.csv",
#     # "/struct/mahamid/Irene/yeast/healthy/180426/043/2bin/fas/motl/motl_job003.csv",
#     # "/struct/mahamid/Irene/yeast/healthy/180426/045/2bin/fas/motl/motl_job003.csv",
# ]
#
# output_dir_paths = [
#     "/struct/mahamid/Irene/yeast/healthy/180426/004/2bin/fas/motl",
#     "/struct/mahamid/Irene/yeast/healthy/180426/005/2bin/fas/motl",
#     "/struct/mahamid/Irene/yeast/healthy/180426/021/2bin/fas/motl",
#     "/struct/mahamid/Irene/yeast/healthy/180426/024/2bin/fas/motl",
#     "/struct/mahamid/Irene/yeast/healthy/180426/026/2bin/fas/motl",
#     "/struct/mahamid/Irene/yeast/healthy/180426/027/2bin/fas/motl",
#     "/struct/mahamid/Irene/yeast/healthy/180426/028/2bin/fas/motl",
#     "/struct/mahamid/Irene/yeast/healthy/180426/029/2bin/fas/motl",
#     "/struct/mahamid/Irene/yeast/healthy/180426/030/2bin/fas/motl",
#     "/struct/mahamid/Irene/yeast/healthy/180426/034/2bin/fas/motl",
#     "/struct/mahamid/Irene/yeast/healthy/180426/037/2bin/fas/motl",
#     "/struct/mahamid/Irene/yeast/healthy/180426/041/2bin/fas/motl",
#     "/struct/mahamid/Irene/yeast/healthy/180426/043/2bin/fas/motl",
#     "/struct/mahamid/Irene/yeast/healthy/180426/045/2bin/fas/motl",
# ]
#
# dataset_shapes = [
#     [960, 927, 1000],
#     [960, 927, 1000],
#     [960, 927, 1000],
#     [960, 927, 1000],
#     [960, 927, 1000],
#     [960, 928, 500],
#     [960, 928, 500],
#     [960, 928, 500],
#     [960, 928, 500],
#     [960, 928, 500],
#     [960, 928, 500],
#     [960, 928, 500],
#     [960, 928, 500],
#     [960, 928, 500],
# ]
# import os
#
# import numpy as np
#
# from file_actions.readers.motl import load_motl_as_df
#
# shifts_xyz = [
#     [0, 0, 0],
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 0, 1],
#     [1, 1, 0],
#     [1, 0, 1],
#     [0, 1, 1],
#     [1, 1, 1],
# ]
#
#
#
# prefixes = [
#     "part1",
#     "part2",
#     "part3",
#     "part4",
#     "part5",
#     "part6",
#     "part7",
#     "part8",
# ]
# tol = [24, 24, 24]
# for motl_path, dataset_shape, output_dir_path in zip(path_to_motls[:1],
#                                                      dataset_shapes,
#                                                      output_dir_paths):
#     motl = load_motl_as_df(motl_path)
#     print(motl.shape)
#     print("original 4bin dataset shape", dataset_shape)
#     half_z_shape = int(dataset_shape[2]/4)
#     global_xyz_shift = np.array([0, 0, half_z_shape])
#     for shift, prefix in zip(shifts_xyz, prefixes):
#         init_corner = np.array(shift) * np.array(dataset_shape)
#         init_corner_with_tolerance = init_corner - np.array(tol)
#         end_corner = init_corner_with_tolerance + np.array(
#             dataset_shape) + np.array(tol)
#         submotl = motl[
#             (motl["x"] < end_corner[0]) &
#             (motl["y"] < end_corner[1]) &
#             (motl["z"] < end_corner[2])
#             ]
#         submotl = submotl[
#             (init_corner_with_tolerance[0] <= submotl["x"]) &
#             (init_corner_with_tolerance[1] <= submotl["y"]) &
#             (init_corner_with_tolerance[2] <= submotl["z"])
#             ]
#         print(submotl.shape)
#         init_corner_str = ""
#         for entry in init_corner:
#             init_corner_str += "_" + str(entry)
#         # print(init_corner_str)
#         submotl_name = prefix + "_submotl_tol24" + init_corner_str + ".csv"
#         output_motl = os.path.join(output_dir_path, submotl_name)
#         print(output_motl)
#         submotl.to_csv(path_or_buf=output_motl, index=False, header=None)

# import os
#
# import h5py
# import numpy as np
#
# from tomogram_utils.coordinates_toolbox.subtomos import get_coord_from_name
# from file_actions.writers.mrc import write_mrc_dataset
# path_to_dirs = [
#     # "/struct/mahamid/Irene/scratch/trueba/3d-cnn/cnn_evaluation/test_partitions/2bin/180426/004/",
#     # "/struct/mahamid/Irene/scratch/trueba/3d-cnn/cnn_evaluation/test_partitions/2bin/180426/005/",
#     # # "/struct/mahamid/Irene/scratch/trueba/3d-cnn/cnn_evaluation/test_partitions/2bin/180426/021/",
#     # # "/struct/mahamid/Irene/scratch/trueba/3d-cnn/cnn_evaluation/test_partitions/2bin/180426/024/",
#     # "/struct/mahamid/Irene/scratch/trueba/3d-cnn/cnn_evaluation/test_partitions/2bin/180426/026/",
#     "/struct/mahamid/Irene/scratch/trueba/3d-cnn/cnn_evaluation/test_partitions/2bin/180426/027/",
#     "/struct/mahamid/Irene/scratch/trueba/3d-cnn/cnn_evaluation/test_partitions/2bin/180426/028/",
#     "/struct/mahamid/Irene/scratch/trueba/3d-cnn/cnn_evaluation/test_partitions/2bin/180426/029/",
#     "/struct/mahamid/Irene/scratch/trueba/3d-cnn/cnn_evaluation/test_partitions/2bin/180426/030/",
#     "/struct/mahamid/Irene/scratch/trueba/3d-cnn/cnn_evaluation/test_partitions/2bin/180426/034/",
#     "/struct/mahamid/Irene/scratch/trueba/3d-cnn/cnn_evaluation/test_partitions/2bin/180426/037/",
#     "/struct/mahamid/Irene/scratch/trueba/3d-cnn/cnn_evaluation/test_partitions/2bin/180426/041/",
#     "/struct/mahamid/Irene/scratch/trueba/3d-cnn/cnn_evaluation/test_partitions/2bin/180426/043/",
#     "/struct/mahamid/Irene/scratch/trueba/3d-cnn/cnn_evaluation/test_partitions/2bin/180426/045/",
# ]
#
# dataset_shapes = [
#     # np.array([960, 927, 1000]),
#     # np.array([960, 927, 1000]),
#     # # np.array([960, 927, 1000]),
#     # # np.array([960, 927, 1000]),
#     # np.array([960, 927, 1000]),
#     np.array([960, 927, 500]),
#     np.array([960, 927, 500]),
#     np.array([960, 927, 500]),
#     np.array([960, 927, 500]),
#     np.array([960, 927, 500]),
#     np.array([960, 927, 500]),
#     np.array([960, 927, 500]),
#     np.array([960, 927, 500]),
#     np.array([960, 927, 500]),
# ]
# dataset_half_shapes = [
#     # np.array([480, 463, 500]),
#     # np.array([480, 463, 500]),
#     # # np.array([480, 463, 500]),
#     # # np.array([480, 463, 500]),
#     # np.array([480, 463, 500]),
#     np.array([480, 463, 250]),
#     np.array([480, 463, 250]),
#     np.array([480, 463, 250]),
#     np.array([480, 463, 250]),
#     np.array([480, 463, 250]),
#     np.array([480, 463, 250]),
#     np.array([480, 463, 250]),
#     np.array([480, 463, 250]),
#     np.array([480, 463, 250]),
#     np.array([480, 463, 250]),
# ]
# print(len(dataset_shapes), len(path_to_dirs), len(dataset_half_shapes))
# assert (len(dataset_shapes) == len(path_to_dirs))
#
# prefixes = [
#     "part1",
#     "part2",
#     "part3",
#     "part4",
# ]
#
# shifts = [
#     np.array([0, 0, 0]),
#     np.array([1, 0, 0]),
#     np.array([0, 1, 0]),
#     np.array([0, 0, 1]),
#     np.array([1, 1, 0]),
#     np.array([1, 0, 1]),
#     np.array([0, 1, 1]),
#     np.array([1, 1, 1]),
# ]
#
# shapes_dict = {}
# for prefix, shift in zip(prefixes, shifts):
#     shapes_dict[prefix] = shift
#
# for path_to_dir, dataset_half_shape, dataset_shape in zip(path_to_dirs,
#                                                           dataset_half_shapes,
#                                                           dataset_shapes):
#     input_path = os.path.join(path_to_dir, "tomo_partition.h5")
#     with h5py.File(input_path, 'r') as f:
#         subregion_names = list(f['volumes/raw'])
#         for subregion_name in subregion_names:
#             # print(subregion_name)
#             zyx_coord = get_coord_from_name(subregion_name)
#             xyz_coord = list(reversed(zyx_coord))
#             for index in range(3):
#                 if xyz_coord[index] % 2 == 3:
#                     xyz_coord[index] = xyz_coord[index] + 1
#
#             xyz_corner = [coord - shape for coord, shape in
#                           zip(xyz_coord, dataset_half_shape)]
#             # print(xyz_corner)
#             for key in shapes_dict.keys():
#                 value = shapes_dict[key]
#                 real_shift = list(value * dataset_shape)
#                 if xyz_corner == real_shift:
#                     print("key, xyz_corner", key, xyz_corner)
#                     corner_str = ''
#                     for item in xyz_corner:
#                         corner_str += '_' + str(int(item))
#                     output_file_name = key + "_subregion" + corner_str + ".mrc"
#                     output_path = os.path.join(path_to_dir, output_file_name)
#                     # print(output_path)
#                     subregion_h5_path = os.path.join('volumes/raw',
#                                                      subregion_name)
#                     subregion = f[subregion_h5_path][:]
#                     write_mrc_dataset(mrc_path=output_path, array=subregion)
#
#
