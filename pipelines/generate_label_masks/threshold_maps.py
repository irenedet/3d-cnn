from src.python.filereaders.datasets import load_dataset
from src.python.filewriters.h5 import write_dataset_hdf

data_files = [
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00200_70S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00201_70S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00203_70S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00204_70S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00205_70S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00206_70S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00207_70S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00208_70S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00240_70S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00241_70S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00242_70S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00243_70S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00244_70S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00245_70S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00246_70S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00247_70S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00248_70S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00249_70S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00250_70S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00251_70S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00200_50S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00201_50S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00203_50S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00204_50S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00205_50S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00206_50S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00207_50S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00208_50S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00240_50S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00241_50S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00242_50S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00243_50S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00244_50S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00245_50S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00246_50S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00247_50S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00248_50S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00249_50S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00250_50S_mask.em",
    "/struct/mahamid/Liang/tomogram/3D_CNN/network_training_3subclasses/data/00251_50S_mask.em",
]
output_paths = [
    "/struct/mahamid/Irene/liang_data/data_multiclass/00200_70S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00201_70S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00203_70S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00204_70S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00205_70S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00206_70S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00207_70S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00208_70S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00240_70S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00241_70S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00242_70S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00243_70S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00244_70S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00245_70S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00246_70S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00247_70S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00248_70S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00249_70S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00250_70S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00251_70S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00200_50S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00201_50S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00203_50S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00204_50S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00205_50S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00206_50S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00207_50S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00208_50S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00240_50S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00241_50S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00242_50S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00243_50S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00244_50S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00245_50S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00246_50S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00247_50S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00248_50S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00249_50S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00250_50S_mask.hdf",
    "/struct/mahamid/Irene/liang_data/data_multiclass/00251_50S_mask.hdf",
]
# to define:
# thresholds = [
#     116,
# ]

# lamellas = [
#     # "/struct/mahamid/Irene/yeast/ED/181119/002/lamellamask.hdf",
#     # "/struct/mahamid/Irene/yeast/ED/181119/030/lamellamask.hdf",
#     # "/struct/mahamid/Irene/yeast/ED/181126/002/lamellamask.hdf",
#     # "/struct/mahamid/Irene/yeast/ED/181126/012/lamellamask.hdf",
#     # "/struct/mahamid/Irene/yeast/ED/181126/025/lamellamask.hdf",
#     # "/struct/mahamid/Irene/yeast/ED/190301/005/lamellamask.hdf",
#     "/struct/mahamid/twalther/Processing/190301/031/TM/031_lamellamask.em",
#     "/struct/mahamid/twalther/Processing/190301/033/TM/033_lamellamask.em",
# ]

# print(len(data_files), len(thresholds))
assert len(data_files) == len(output_paths)
# assert len(data_files) == len(thresholds)
value = 1

# lamella_mask = "/struct/mahamid/Irene/liang_data/lamella.hdf"

for hdf_file_path, output_file_path in zip(data_files, output_paths):
    print("Binarizing ", hdf_file_path)
    tomo_data = load_dataset(path_to_dataset=hdf_file_path)
    thresholded_dataset = value * (tomo_data > 0.5)
    write_dataset_hdf(output_path=output_file_path,
                      tomo_data=thresholded_dataset)

