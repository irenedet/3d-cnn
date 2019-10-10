from src.python.filereaders.datasets import load_dataset
from src.python.filewriters.h5 import write_dataset_hdf

data_files = [
    "/struct/mahamid/Irene/yeast/healthy/180413/006/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180413/007/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180426/005/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180426/006/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180426/008/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180426/014/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180426/027/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180426/037/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180426/038/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180426/040/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180426/043/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180711/005/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180711/006/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180711/007/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180711/012/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180711/017/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180711/022/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180713/002/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180713/005/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180713/007/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180713/015/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180713/018/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180713/020/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180713/025/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180713/027/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180713/031/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180713/035/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180713/037/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180713/039/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180713/041/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180713/043/clean_masks/npc_cylindrical_mask.em",
    "/struct/mahamid/Irene/yeast/healthy/180713/050/clean_masks/npc_cylindrical_mask.em",
]

output_paths = [
    "/struct/mahamid/Irene/yeast/healthy/180413/006/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180413/007/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180426/005/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180426/006/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180426/008/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180426/014/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180426/027/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180426/037/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180426/038/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180426/040/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180426/043/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180711/005/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180711/006/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180711/007/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180711/012/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180711/017/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180711/022/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180713/002/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180713/005/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180713/007/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180713/015/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180713/018/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180713/020/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180713/025/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180713/027/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180713/031/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180713/035/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180713/037/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180713/039/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180713/041/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180713/043/clean_masks/npc_cylindrical_mask_bin.hdf",
    "/struct/mahamid/Irene/yeast/healthy/180713/050/clean_masks/npc_cylindrical_mask_bin.hdf",
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

print(len(data_files), len(output_paths))
assert len(data_files) == len(output_paths)
value = 1
threshold = 0.75
# lamella_mask = "/struct/mahamid/Irene/liang_data/lamella.hdf"
# path_to_mask = "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/Templates/FAS_mask_4b_b32_r10.em"
for input_file_path, output_file_path in zip(data_files, output_paths):
    print("Binarizing ", input_file_path)
    # mask = load_dataset(path_to_dataset=path_to_mask)
    tomo_data = load_dataset(path_to_dataset=input_file_path)
    # tomo_data *= mask
    thresholded_dataset = value * (tomo_data >= threshold)
    write_dataset_hdf(output_path=output_file_path,
                      tomo_data=thresholded_dataset)


