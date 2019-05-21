from os.path import join
from src.python.filereaders.hdf import _load_hdf_dataset
from src.python.filewriters.h5 import write_dataset_hdf
from os import makedirs

data_files = [
    # "/struct/mahamid/Irene/yeast/vpp/180426_004/ribos/spherical_clean_mask_corrected.hdf",
    # "/struct/mahamid/Irene/yeast/vpp/180426_005/ribos/spherical_clean_mask.hdf",
    # "/struct/mahamid/Irene/yeast/vpp/180426_021/ribos/spherical_clean_mask.hdf",
    # "/struct/mahamid/Irene/yeast/vpp/180426_024/ribos/spherical_clean_mask.hdf",
    # "/struct/mahamid/Irene/yeast/vpp/180711_003/ribos/spherical_clean_mask.hdf",
    # "/struct/mahamid/Irene/yeast/vpp/180711_004/ribos/spherical_clean_mask.hdf",
    # "/struct/mahamid/Irene/yeast/vpp/180711_005/ribos/spherical_clean_mask.hdf",
    # "/struct/mahamid/Irene/yeast/vpp/180711_018/ribos/spherical_clean_mask.hdf",
    "/struct/mahamid/Irene/yeast/vpp/180713_027/ribos/spherical_clean_mask.hdf",
    ]
output_paths = [
    # "/struct/mahamid/Irene/yeast/vpp/180426_004/ribos",
    # "/struct/mahamid/Irene/yeast/vpp/180426_005/ribos",
    # "/struct/mahamid/Irene/yeast/vpp/180426_021/ribos",
    # "/struct/mahamid/Irene/yeast/vpp/180426_024/ribos",
    # "/struct/mahamid/Irene/yeast/vpp/180711_003/ribos",
    # "/struct/mahamid/Irene/yeast/vpp/180711_004/ribos",
    # "/struct/mahamid/Irene/yeast/vpp/180711_005/ribos",
    # "/struct/mahamid/Irene/yeast/vpp/180711_018/ribos",
    "/struct/mahamid/Irene/yeast/vpp/180713_027/ribos",
    ]
# output_dir = "/struct/mahamid/Irene/yeast/ribosomes/180426_004/fas_corrected/"
# output_shape = (1000, 927, 927)
threshold = 0  # fas: 0.296
value = 1

print(list(zip(data_files, output_paths)))
for hdf_file_path, output_dir in zip(data_files, output_paths):
    makedirs(name=output_dir, exist_ok=True)
    output_file_path = join(output_dir, "spherical_binarized_mask.hdf")
    print(output_file_path)
    print("Binarizing ", hdf_file_path, " at threshold", threshold)
    tomo_data = _load_hdf_dataset(hdf_file_path=hdf_file_path)
    output_dataset = value * (tomo_data > threshold)
    write_dataset_hdf(output_path=output_file_path, tomo_data=output_dataset)
