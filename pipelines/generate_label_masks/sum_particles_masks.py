from os.path import join
import numpy as np
from src.python.filereaders.hdf import _load_hdf_dataset
from src.python.filewriters.h5 import write_dataset_hdf


data_files = ["/struct/mahamid/Irene/yeast/ribosomes/180426_005/particles_mask.hdf",
              "/struct/mahamid/Irene/yeast/ribosomes/180426_005/corrected/particles_mask.hdf"]

output_path = "/struct/mahamid/Irene/yeast/ribosomes/180426_005/"

output_shape = (1000, 928, 928)

output_dataset = np.zeros(output_shape)
for n, hdf_file_path in enumerate(data_files):
    value = 1
    print("label of ", hdf_file_path, "is", value)
    data_n = _load_hdf_dataset(hdf_file_path=hdf_file_path)
    output_dataset += value*(data_n > 0)

output_file_path = join(output_path, "particles_corrected.hdf")

write_dataset_hdf(output_path=output_file_path, tomo_data=output_dataset)
