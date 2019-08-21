from src.python.filereaders.datasets import load_dataset
from src.python.filewriters.h5 import write_dataset_hdf

data_files = [
    "/struct/mahamid/Irene/yeast/ED/181119/002/memb/tomosegresult.hdf",
    "/struct/mahamid/Irene/yeast/ED/181119/030/memb/tomosegresult.hdf",
    "/struct/mahamid/Irene/yeast/ED/181126/002/memb/tomosegresult.hdf",
    "/struct/mahamid/Irene/yeast/ED/181126/012/memb/tomosegresult.hdf",
    "/struct/mahamid/Irene/yeast/ED/181126/025/memb/tomosegresult.hdf",
]
output_paths = [
    "/struct/mahamid/Irene/yeast/ED/181119/002/memb/binary_tomosegresult.hdf",
    "/struct/mahamid/Irene/yeast/ED/181119/030/memb/binary_tomosegresult.hdf",
    "/struct/mahamid/Irene/yeast/ED/181126/002/memb/binary_tomosegresult.hdf",
    "/struct/mahamid/Irene/yeast/ED/181126/012/memb/binary_tomosegresult.hdf",
    "/struct/mahamid/Irene/yeast/ED/181126/025/memb/binary_tomosegresult.hdf",
]  # to define:
thresholds = [
    38.6,
    39.6,
    77.4,
    92,
    100,
]

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
# assert len(data_files) == len(output_paths)
# assert len(data_files) == len(thresholds)
value = 1

# lamella_mask = "/struct/mahamid/Irene/liang_data/lamella.hdf"
for hdf_file_path, output_file_path, threshold in zip(data_files,
                                                      output_paths,
                                                      thresholds):
    print("Binarizing ", hdf_file_path, " at threshold", threshold)
    tomo_data = load_dataset(path_to_dataset=hdf_file_path)
    thresholded_dataset = value * (tomo_data > threshold)
    write_dataset_hdf(output_path=output_file_path,
                      tomo_data=thresholded_dataset)
