from os.path import join
import numpy as np
from src.python.filereaders.hdf import _load_hdf_dataset
from src.python.filewriters.h5 import write_dataset_hdf

<<<<<<< HEAD
data_files = ["/struct/mahamid/Irene/yeast/180426/004/training/dice_multi_class/ribos.hdf",
              "/struct/mahamid/Irene/yeast/180426/004/training/dice_multi_class/corrected.hdf"]

output_path = "/struct/mahamid/Irene/yeast/180426/004/training/dice_multi_class"
=======
data_files = ["/scratch/trueba/3d-cnn/clean/180426_004/fas_clean_mask.hdf",
              "/scratch/trueba/3d-cnn/clean/180426_004/clean_mask.hdf"]

output_path = "/scratch/trueba/3d-cnn/clean/180426_004/"
>>>>>>> a989e851b8c49f42fa934d68991d6e56448b2c0f
output_shape = (221, 928, 928)

output_dataset = np.zeros(output_shape)
for n, hdf_file_path in enumerate(data_files):
    print("label of ", hdf_file_path, "is", n+1)
    data_n = _load_hdf_dataset(hdf_file_path=hdf_file_path)
<<<<<<< HEAD
    output_dataset += 1*(data_n > 0)

output_file_path = join(output_path, "ribos_corrected.hdf")
=======
    output_dataset += (n+1)*(data_n > 0)

output_file_path = join(output_path, "fas_ribos_clean_mask.hdf")
>>>>>>> a989e851b8c49f42fa934d68991d6e56448b2c0f
write_dataset_hdf(output_path=output_file_path, tomo_data=output_dataset)
