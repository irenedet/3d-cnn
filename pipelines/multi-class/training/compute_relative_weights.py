from os.path import join
import numpy as np
from os import makedirs

from src.python.filereaders.hdf import _load_hdf_dataset
from src.python.filewriters.h5 import write_dataset_hdf

hdf_file_path = "/struct/mahamid/Irene/yeast/180426/004/training/multi_class/ribos_2_corrected_2_fas_3_memb_1_mask.hdf"
data = np.array(_load_hdf_dataset(hdf_file_path=hdf_file_path))

print("done reading data...")

values = [0, 1, 2, 3]

count_list = []

for value in values:
    print(value)
    count = np.count_nonzero(1*np.where(data == value))
    count_list += [count]

weights = [count_list[2]/count for count in count_list]

print(count_list)
print(weights)