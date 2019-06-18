from os.path import join
from src.python.filereaders.hdf import _load_hdf_dataset
from src.python.filereaders.datasets import load_dataset
from src.python.filewriters.h5 import write_dataset_hdf
from os import makedirs
import numpy as np

data_files = [
    "/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/Retrain_retrained_D4_IF8_NA_except_180711_003/double_eman_filt/181119_002/memb.hdf",
    "/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/Retrain_retrained_D4_IF8_NA_except_180711_003/double_eman_filt/181119_030/memb.hdf",
    "/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/Retrain_retrained_D4_IF8_NA_except_180711_003/double_eman_filt/181126_002/memb.hdf",
    "/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/Retrain_retrained_D4_IF8_NA_except_180711_003/double_eman_filt/181126_012/memb.hdf",
    "/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/Retrain_retrained_D4_IF8_NA_except_180711_003/double_eman_filt/181126_025/memb.hdf",
    "/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/Retrain_retrained_D4_IF8_NA_except_180711_003/double_eman_filt/190301_005/memb.hdf",
]
output_paths = [
    "/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/Retrain_retrained_D4_IF8_NA_except_180711_003/double_eman_filt/181119_002/thr_binary_memb.hdf",
    "/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/Retrain_retrained_D4_IF8_NA_except_180711_003/double_eman_filt/181119_030/thr_binary_memb.hdf",
    "/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/Retrain_retrained_D4_IF8_NA_except_180711_003/double_eman_filt/181126_002/thr_binary_memb.hdf",
    "/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/Retrain_retrained_D4_IF8_NA_except_180711_003/double_eman_filt/181126_012/thr_binary_memb.hdf",
    "/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/Retrain_retrained_D4_IF8_NA_except_180711_003/double_eman_filt/181126_025/thr_binary_memb.hdf",
    "/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/Retrain_retrained_D4_IF8_NA_except_180711_003/double_eman_filt/190301_005/thr_binary_memb.hdf",
]

thresholds = [
    0,
    0,
    0,
    0,
    0,
    0,
]

lamellas = [
    "/struct/mahamid/Irene/yeast/ED/181119_002/002_lamellamask.hdf",
    "/struct/mahamid/Irene/yeast/ED/181119_030/030_lamellamask.hdf",
    "/struct/mahamid/Irene/yeast/ED/181126_002/002_lamellamask.hdf",
    "/struct/mahamid/Irene/yeast/ED/181126_012/012_lamellamask.hdf",
    "/struct/mahamid/Irene/yeast/ED/181126_025/025_lamellamask.hdf",
    "/struct/mahamid/Irene/yeast/ED/190301_005/005_lamellamask.hdf",
]

print(len(data_files), len(thresholds))
assert len(data_files) == len(output_paths)
assert len(data_files) == len(thresholds)
value = 1

print(list(zip(data_files, output_paths, thresholds)))
for hdf_file_path, output_file_path, threshold, lamella_mask in zip(data_files,
                                                                    output_paths,
                                                                    thresholds,
                                                                    lamellas):
    print("Binarizing ", hdf_file_path, " at threshold", threshold)
    tomo_data = load_dataset(path_to_dataset=hdf_file_path)
    thresholded_dataset = value * (tomo_data > threshold)
    lamella_dataset = _load_hdf_dataset(lamella_mask)
    min_shape = [np.min([sh1, sh2]) for sh1, sh2 in
                 zip(lamella_dataset.shape, thresholded_dataset.shape)]
    dimz, dimy, dimx = min_shape
    output_dataset = thresholded_dataset[:dimz, :dimy, :dimx] * lamella_dataset[
                                                                :dimz, :dimy,
                                                                :dimx]
    write_dataset_hdf(output_path=output_file_path, tomo_data=output_dataset)
