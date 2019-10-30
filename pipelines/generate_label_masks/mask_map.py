from os.path import join
from src.python.filereaders.hdf import _load_hdf_dataset
from src.python.filewriters.h5 import write_dataset_hdf
from os import makedirs

data_files = [
    "/struct/mahamid/Irene/yeast/ED/181119/030/memb/binary_tomosegresult.hdf",
]
output_paths = [
    "/struct/mahamid/Irene/yeast/ED/181119/030/memb/binary_tomosegresult_no_border.hdf",
]

assert len(data_files) == len(output_paths)
# assert len(data_files) == len(masks)
import numpy as np

xy_border = 10
print(list(zip(data_files, output_paths)))
for hdf_file_path, output_file_path in zip(data_files, output_paths):
    tomo_data = _load_hdf_dataset(hdf_file_path=hdf_file_path)
    # mask_data = _load_hdf_dataset(hdf_file_path=mask_path)
    tomo_data = np.array(tomo_data)
    shz, shy, shx = tomo_data.shape
    output_dataset = np.zeros(tomo_data.shape)
    output_dataset[xy_border:shz - xy_border, xy_border:shy - xy_border,
    xy_border:shx - xy_border] = tomo_data[xy_border:shz - xy_border,
                                 xy_border:shy - xy_border,
                                 xy_border:shx - xy_border]
    write_dataset_hdf(output_path=output_file_path, tomo_data=output_dataset)
