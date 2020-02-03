import h5py

original_paths = [
    "/struct/mahamid/Irene/yeast/ED/181126/002/raw_tomo.hdf",
    "/struct/mahamid/Irene/yeast/ED/181126/002/clean_masks/ribo_sph_mask.hdf",
]

destination_paths = [
    "/struct/mahamid/Irene/yeast/ED/181126/002/raw_tomo.h5",
    "/struct/mahamid/Irene/yeast/ED/181126/002/clean_masks/ribo_sph_mask.h5",
]

for hdf_path, h5_path in zip(original_paths, destination_paths):
    print("starting with", hdf_path)
    with h5py.File(hdf_path, 'r') as hdf_f:
        dataset = hdf_f["MDF/images/0/image"][:]
        # print(dataset.shape)

    with h5py.File(h5_path, 'w') as h5_f:
        h5_f.create_dataset("dataset", data=dataset)

print("finished!")

