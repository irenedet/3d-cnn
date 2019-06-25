from os.path import join
from src.python.filereaders.hdf import _load_hdf_dataset
from src.python.filereaders.datasets import load_dataset
from src.python.filewriters.h5 import write_dataset_hdf
from os import makedirs
import numpy as np

data_files = [
    # "/struct/mahamid/Irene/yeast/ED/181119/002/memb/tomosegresult.hdf",
    # "/struct/mahamid/Irene/yeast/ED/181119/030/memb/tomosegresult.hdf",
    # "/struct/mahamid/Irene/yeast/ED/181126/002/memb/tomosegresult.hdf",
    # "/struct/mahamid/Irene/yeast/ED/181126/012/memb/tomosegresult.hdf",
    # "/struct/mahamid/Irene/yeast/ED/181126/025/memb/tomosegresult.hdf",
    # "/struct/mahamid/Irene/yeast/ED/190301/005/memb/tomosegresult.hdf",
    "/struct/mahamid/Irene/yeast/ED/190301/031/memb/tomosegresult.hdf",
    "/struct/mahamid/Irene/yeast/ED/190301/033/memb/tomosegresult.hdf",
]
output_paths = [
    # "/struct/mahamid/Irene/yeast/ED/181119/002/memb/binary_tomosegresult.hdf",
    # "/struct/mahamid/Irene/yeast/ED/181119/030/memb/binary_tomosegresult.hdf",
    # "/struct/mahamid/Irene/yeast/ED/181126/002/memb/binary_tomosegresult.hdf",
    # "/struct/mahamid/Irene/yeast/ED/181126/012/memb/binary_tomosegresult.hdf",
    # "/struct/mahamid/Irene/yeast/ED/181126/025/memb/binary_tomosegresult.hdf",
    # "/struct/mahamid/Irene/yeast/ED/190301/005/memb/binary_tomosegresult.hdf",
    "/struct/mahamid/Irene/yeast/ED/190301/031/memb/binary_tomosegresult.hdf",
    "/struct/mahamid/Irene/yeast/ED/190301/033/memb/binary_tomosegresult.hdf",
]

#to define:
thresholds = [
    0,
    0,
    # 0,
    # 0,
    # 0,
    # 0,
    # 0,
    # 0,
]

lamellas = [
    # "/struct/mahamid/Irene/yeast/ED/181119/002/lamellamask.hdf",
    # "/struct/mahamid/Irene/yeast/ED/181119/030/lamellamask.hdf",
    # "/struct/mahamid/Irene/yeast/ED/181126/002/lamellamask.hdf",
    # "/struct/mahamid/Irene/yeast/ED/181126/012/lamellamask.hdf",
    # "/struct/mahamid/Irene/yeast/ED/181126/025/lamellamask.hdf",
    # "/struct/mahamid/Irene/yeast/ED/190301/005/lamellamask.hdf",
    "/struct/mahamid/twalther/Processing/190301/031/TM/031_lamellamask.em",
    "/struct/mahamid/twalther/Processing/190301/033/TM/033_lamellamask.em",
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
    lamella_dataset = load_dataset(path_to_dataset=lamella_mask)
    min_shape = [np.min([sh1, sh2]) for sh1, sh2 in
                 zip(lamella_dataset.shape, thresholded_dataset.shape)]
    dimz, dimy, dimx = min_shape
    output_dataset = thresholded_dataset[:dimz, :dimy, :dimx] * lamella_dataset[
                                                                :dimz, :dimy,
                                                                :dimx]
    write_dataset_hdf(output_path=output_file_path, tomo_data=output_dataset)
