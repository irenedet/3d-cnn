from os.path import join
import numpy as np
from src.python.filereaders.hdf import _load_hdf_dataset
from src.python.filewriters.h5 import write_dataset_hdf

data_directory = "/struct/mahamid/Irene/shrec/0"
output_path = "/struct/mahamid/Irene/shrec/0"
output_shape = (512, 512, 512)

output_dataset = np.zeros(output_shape)
for n in range(12):
    hdf_file_name = str(n + 1) + "_mask_0.hdf"
    hdf_file_path = join(data_directory, hdf_file_name)
    data_n = _load_hdf_dataset(hdf_file_path=hdf_file_path)
    min_n = np.min(data_n)
    print("In ", hdf_file_name, " np.min(data_n) = ", min_n)
    # output_dataset += 1 * (data_n > 0)
    output_dataset += (n+1)*(data_n > 0)

output_file_path = join(output_path, "all_classes_differentiated_mask_0.hdf")
write_dataset_hdf(output_path=output_file_path, tomo_data=output_dataset)
