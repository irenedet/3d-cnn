from os.path import join
from src.python.filereaders.hdf import _load_hdf_dataset
from src.python.filewriters.h5 import write_dataset_hdf
from os import makedirs

data_files = [
    # "/struct/mahamid/Irene/yeast/vpp/180426_004/memb/binarized_mask_sq.hdf",
    # "/struct/mahamid/Irene/yeast/vpp/180426_005/memb/binarized_mask.hdf",
    "/struct/mahamid/Irene/yeast/vpp/180426_021/memb/binarized_mask.hdf",
    "/struct/mahamid/Irene/yeast/vpp/180426_024/memb/binarized_mask.hdf",
    "/struct/mahamid/Irene/yeast/vpp/180711_003/memb/binarized_mask.hdf",
    "/struct/mahamid/Irene/yeast/vpp/180711_004/memb/binarized_mask.hdf",
    "/struct/mahamid/Irene/yeast/vpp/180711_005/memb/binarized_mask.hdf",
    "/struct/mahamid/Irene/yeast/vpp/180711_018/memb/binarized_mask.hdf",
    "/struct/mahamid/Irene/yeast/vpp/180713_027/memb/binarized_mask.hdf",
]
output_paths = [
    # "/struct/mahamid/Irene/yeast/vpp/180426_004/memb/binarized_lamella_mask.hdf",
    # "/struct/mahamid/Irene/yeast/vpp/180426_005/memb/binarized_lamella_mask.hdf",
    "/struct/mahamid/Irene/yeast/vpp/180426_021/memb/binarized_lamella_mask.hdf",
    "/struct/mahamid/Irene/yeast/vpp/180426_024/memb/binarized_lamella_mask.hdf",
    "/struct/mahamid/Irene/yeast/vpp/180711_003/memb/binarized_lamella_mask.hdf",
    "/struct/mahamid/Irene/yeast/vpp/180711_004/memb/binarized_lamella_mask.hdf",
    "/struct/mahamid/Irene/yeast/vpp/180711_005/memb/binarized_lamella_mask.hdf",
    "/struct/mahamid/Irene/yeast/vpp/180711_018/memb/binarized_lamella_mask.hdf",
    "/struct/mahamid/Irene/yeast/vpp/180713_027/memb/binarized_lamella_mask.hdf",
]

masks = [
    # "/scratch/trueba/3d-cnn/clean/180426_004/004_lamellamask.hdf",
    # "/scratch/trueba/3d-cnn/clean/180426_005/005_lamellamask.hdf",
    "/scratch/trueba/3d-cnn/clean/180426_021/021_lamellamask.hdf",
    "/scratch/trueba/3d-cnn/clean/180426_024/024_lamellamask.hdf",
    "/scratch/trueba/3d-cnn/clean/180711_003/003_lamellamask.hdf",
    "/scratch/trueba/3d-cnn/clean/180711_004/004_lamellamask.hdf",
    "/scratch/trueba/3d-cnn/clean/180711_005/005_lamellamask.hdf",
    "/scratch/trueba/3d-cnn/clean/180711_018/018_lamellamask.hdf",
    "/scratch/trueba/3d-cnn/clean/180713_027/027_lamellamask.hdf",
]

assert len(data_files) == len(output_paths)
assert len(data_files) == len(masks)
import numpy as np

print(list(zip(data_files, output_paths, masks)))
for hdf_file_path, output_file_path, mask_path in zip(data_files, output_paths,
                                                      masks):
    tomo_data = _load_hdf_dataset(hdf_file_path=hdf_file_path)
    mask_data = _load_hdf_dataset(hdf_file_path=mask_path)
    tomo_data = np.array(tomo_data)
    mask_data = np.array(mask_data)
    output_dataset = np.zeros(tomo_data.shape)
    min_shape = [np.min([sh1, sh2]) for sh1, sh2 in
                 zip(mask_data.shape, tomo_data.shape)]
    minz, miny, minx = min_shape
    print("min_shape = ", min_shape)
    output_dataset[:minz, :miny, :minx] = output_dataset[:minz, :miny,
                                          :minx] * tomo_data[:minz, :miny,
                                                   :minx]
    write_dataset_hdf(output_path=output_file_path, tomo_data=output_dataset)
