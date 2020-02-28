import numpy as np

from file_actions.readers.hdf import _load_hdf_dataset

hdf_file_path = "/struct/mahamid/Irene/yeast/180426/004/training/multi_class/ribos_2_corrected_2_fas_3_memb_1_mask.hdf"
data = np.array(_load_hdf_dataset(hdf_file_path=hdf_file_path))

print("done reading data...")

values = [0, 1, 2, 3]

count_list = []

for value in values:
    print(value)
    count = np.count_nonzero(1*np.where(data == value))
    count_list += [count]

total_counts = np.sum(count_list)
weights = [count/total_counts for count in count_list]

print(count_list)
print(weights)