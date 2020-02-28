from os.path import join
import numpy as np
from file_actions.readers.hdf import _load_hdf_dataset
from file_actions.writers.h5 import write_dataset_hdf


output_path = "/struct/mahamid/Irene/yeast/180426/004/training/multi_class"
output_shape = (221, 928, 928)

masks_list = [
    {'label': 'ribos',
     'file': "/scratch/trueba/3d-cnn/clean/180426_004/clean_mask.hdf",
     'threshold': 1},
    {'label': 'fas',
     'file': "/scratch/trueba/3d-cnn/clean/180426_004/fas_clean_mask_subregion.hdf",
     'threshold': 0.25},
    {'label': 'memb',
     'file': "/home/papalotl/Sara_Goetz/180426/004/memb/tomosegresult.hdf",
     'threshold': 4},
]

output_dataset = np.zeros(output_shape)
hdf_mask_name = ""
for n, mask_dict in enumerate(masks_list):
    label, hdf_file_path, threshold = mask_dict['label'], mask_dict['file'], \
                                      mask_dict['threshold']
    data = _load_hdf_dataset(hdf_file_path=hdf_file_path)
    data_min = np.min(data)
    print("In ", label, "mask, np.min(data) = ", data_min)
    output_dataset += (n + 1) * (data >= threshold)
    hdf_mask_name += label + "_"

hdf_mask_name += "mask.hdf"

output_file_path = join(output_path, hdf_mask_name)
write_dataset_hdf(output_path=output_file_path, tomo_data=output_dataset)
