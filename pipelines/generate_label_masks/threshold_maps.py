from os.path import join
from src.python.filereaders.hdf import _load_hdf_dataset
from src.python.filewriters.h5 import write_dataset_hdf
from os import makedirs

data_files = [
    "/struct/mahamid/Irene/NPC/01/clean/cylindrical_mask.hdf",
    "/struct/mahamid/Irene/NPC/02/clean/cylindrical_mask.hdf",
    "/struct/mahamid/Irene/NPC/03/clean/cylindrical_mask.hdf",
    "/struct/mahamid/Irene/NPC/04/clean/cylindrical_mask.hdf",
    "/struct/mahamid/Irene/NPC/05/clean/cylindrical_mask.hdf",
    "/struct/mahamid/Irene/NPC/06/clean/cylindrical_mask.hdf",
    "/struct/mahamid/Irene/NPC/10/clean/cylindrical_mask.hdf",
    "/struct/mahamid/Irene/NPC/11/clean/cylindrical_mask.hdf",
]
output_paths = [
    "/struct/mahamid/Irene/NPC/01/clean/binarized_cylindrical_mask.hdf",
    "/struct/mahamid/Irene/NPC/02/clean/binarized_cylindrical_mask.hdf",
    "/struct/mahamid/Irene/NPC/03/clean/binarized_cylindrical_mask.hdf",
    "/struct/mahamid/Irene/NPC/04/clean/binarized_cylindrical_mask.hdf",
    "/struct/mahamid/Irene/NPC/05/clean/binarized_cylindrical_mask.hdf",
    "/struct/mahamid/Irene/NPC/06/clean/binarized_cylindrical_mask.hdf",
    "/struct/mahamid/Irene/NPC/10/clean/binarized_cylindrical_mask.hdf",
    "/struct/mahamid/Irene/NPC/11/clean/binarized_cylindrical_mask.hdf",
]

thresholds = [
    .2,
    .2,
    .2,
    .2,
    .2,
    .2,
    .2,
    .2,
]

assert len(data_files) == len(output_paths)
assert len(data_files) == len(thresholds)
value = 1

print(list(zip(data_files, output_paths, thresholds)))
for hdf_file_path, output_file_path, threshold in zip(data_files, output_paths,
                                                      thresholds):
    print("Binarizing ", hdf_file_path, " at threshold", threshold)
    tomo_data = _load_hdf_dataset(hdf_file_path=hdf_file_path)
    output_dataset = value * (tomo_data > threshold)
    write_dataset_hdf(output_path=output_file_path, tomo_data=output_dataset)
