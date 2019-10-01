from src.python.filereaders.datasets import load_dataset
from src.python.filewriters.h5 import write_dataset_hdf

data_files = [
    '/struct/mahamid/Irene/yeast/ED/190301/003/clean_masks/fas_non_sph_mask.em',
    '/struct/mahamid/Irene/yeast/ED/190301/005/clean_masks/fas_non_sph_mask.em',
    '/struct/mahamid/Irene/yeast/ED/190301/009/clean_masks/fas_non_sph_mask.em',
    '/struct/mahamid/Irene/yeast/ED/190301/012/clean_masks/fas_non_sph_mask.em',
    '/struct/mahamid/Irene/yeast/ED/190301/016/clean_masks/fas_non_sph_mask.em',
    '/struct/mahamid/Irene/yeast/ED/190301/022/clean_masks/fas_non_sph_mask.em',
    '/struct/mahamid/Irene/yeast/ED/190301/028/clean_masks/fas_non_sph_mask.em',
    '/struct/mahamid/Irene/yeast/ED/190301/031/clean_masks/fas_non_sph_mask.em',
    '/struct/mahamid/Irene/yeast/ED/190301/032/clean_masks/fas_non_sph_mask.em',
    '/struct/mahamid/Irene/yeast/ED/190301/033/clean_masks/fas_non_sph_mask.em',
    '/struct/mahamid/Irene/yeast/ED/190301/035/clean_masks/fas_non_sph_mask.em',
    '/struct/mahamid/Irene/yeast/ED/190301/037/clean_masks/fas_non_sph_mask.em',
    '/struct/mahamid/Irene/yeast/ED/190301/043/clean_masks/fas_non_sph_mask.em',
    '/struct/mahamid/Irene/yeast/ED/190301/045/clean_masks/fas_non_sph_mask.em',
]

output_paths = [
    '/struct/mahamid/Irene/yeast/ED/190301/003/clean_masks/fas_non_sph_mask_bin.hdf',
    '/struct/mahamid/Irene/yeast/ED/190301/005/clean_masks/fas_non_sph_mask_bin.hdf',
    '/struct/mahamid/Irene/yeast/ED/190301/009/clean_masks/fas_non_sph_mask_bin.hdf',
    '/struct/mahamid/Irene/yeast/ED/190301/012/clean_masks/fas_non_sph_mask_bin.hdf',
    '/struct/mahamid/Irene/yeast/ED/190301/016/clean_masks/fas_non_sph_mask_bin.hdf',
    '/struct/mahamid/Irene/yeast/ED/190301/022/clean_masks/fas_non_sph_mask_bin.hdf',
    '/struct/mahamid/Irene/yeast/ED/190301/028/clean_masks/fas_non_sph_mask_bin.hdf',
    '/struct/mahamid/Irene/yeast/ED/190301/031/clean_masks/fas_non_sph_mask_bin.hdf',
    '/struct/mahamid/Irene/yeast/ED/190301/032/clean_masks/fas_non_sph_mask_bin.hdf',
    '/struct/mahamid/Irene/yeast/ED/190301/033/clean_masks/fas_non_sph_mask_bin.hdf',
    '/struct/mahamid/Irene/yeast/ED/190301/035/clean_masks/fas_non_sph_mask_bin.hdf',
    '/struct/mahamid/Irene/yeast/ED/190301/037/clean_masks/fas_non_sph_mask_bin.hdf',
    '/struct/mahamid/Irene/yeast/ED/190301/043/clean_masks/fas_non_sph_mask_bin.hdf',
    '/struct/mahamid/Irene/yeast/ED/190301/045/clean_masks/fas_non_sph_mask_bin.hdf', ]
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
threshold = 1
# lamella_mask = "/struct/mahamid/Irene/liang_data/lamella.hdf"
# path_to_mask = "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/Templates/FAS_mask_4b_b32_r10.em"
for hdf_file_path, output_file_path in zip(data_files, output_paths):
    print("Binarizing ", hdf_file_path)
    # mask = load_dataset(path_to_dataset=path_to_mask)
    tomo_data = load_dataset(path_to_dataset=hdf_file_path)
    # tomo_data *= mask
    thresholded_dataset = value * (tomo_data >= threshold)
    write_dataset_hdf(output_path=output_file_path,
                      tomo_data=thresholded_dataset)
