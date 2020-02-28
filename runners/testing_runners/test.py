# import os
#
# from filereaders.datasets import load_tomogram
# from filewriters.h5 import write_dataset_hdf
#
# tomos = [
#     # "190301/003",
#     "190301/005",
#     # "190301/009",
#     # "190301/012",
#     # "190301/016",
#     # "190301/022",
#     # "190301/028",
#     # "190301/031",
#     # "190301/032",
#     # "190301/033",
#     # "190301/035",
#     # "190301/037",
#     # "190301/043",
#     # "190301/045",
# ]
#
# threshold = 0
# value = 1
# for tomo_name in tomos:
#     print("binarizing", tomo_name)
#     path = os.path.join(
#         "/scratch/trueba/3d-cnn/cnn_evaluation/cv_fractions/2/R_false_encoder_dropout_0.2_decoder_dropout_0.2_BN_false_DA_none_shuffle_true_frac_2_ribo_fas_memb__D_2_IF_8",
#         tomo_name)
#     input_data_path = os.path.join(path, "class_2/prediction.hdf")
#     output_path = os.path.join(path, "class_2/binary_prediction.hdf")
#     print(input_data_path)
#     print(output_path)
#     data = load_tomogram(path_to_dataset=input_data_path)
#     # data = data * (data > threshold)
#     data = value * (data > threshold)
#     write_dataset_hdf(output_path=output_path, tomo_data=data)
# print("finished!")
import numpy as np

A = list(range(10))
B = list(range(10, 20))
array = np.array([A, B])
sl = slice(4, 7, None)

print(array[0, sl])

# for s, e in zip(range(10), range(10,20)):
#     print(slice(s,e))
