from os.path import join
import numpy as np
from src.python.filereaders.hdf import _load_hdf_dataset
from src.python.filewriters.h5 import write_dataset_hdf

data_files = ["/scratch/trueba/3d-cnn/clean/180426_004/fas_clean_mask.hdf",
              "/scratch/trueba/3d-cnn/clean/180426_004/clean_mask.hdf"]

output_path = "/scratch/trueba/3d-cnn/clean/180426_004/"
output_shape = (221, 928, 928)

output_dataset = np.zeros(output_shape)
for n, hdf_file_path in enumerate(data_files):
    print("label of ", hdf_file_path, "is", n+1)
    data_n = _load_hdf_dataset(hdf_file_path=hdf_file_path)
    output_dataset += (n+1)*(data_n > 0)

output_file_path = join(output_path, "fas_ribos_clean_mask.hdf")
write_dataset_hdf(output_path=output_file_path, tomo_data=output_dataset)
