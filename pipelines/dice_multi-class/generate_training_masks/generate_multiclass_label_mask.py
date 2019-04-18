from os.path import join
import numpy as np
from os import makedirs

from src.python.filereaders.hdf import _load_hdf_dataset
from src.python.filewriters.h5 import write_dataset_hdf

output_path = "/struct/mahamid/Irene/yeast/180426/004/training/dice_multi_class"
makedirs(name=output_path, exist_ok=True)
output_shape = (221, 928, 928)

masks_list = [
    {'label': 'ribos',
     'file': "/scratch/trueba/3d-cnn/clean/180426_004/clean_mask.hdf",
     'threshold': 1,
     'value': 1},
    {'label': 'corrected',
     'file': "/struct/mahamid/Irene/predictions/180426/004/particles_mask.hdf",
     'threshold': 1,
     'value': 1},
    {'label': 'fas',
     'file': "/scratch/trueba/3d-cnn/clean/180426_004/fas_clean_mask_subregion.hdf",
     'threshold': 0.25,
     'value': 2},
    {'label': 'memb',
     'file': "/scratch/trueba/cnn/004/4bin/cnn/membrane_segmentation.hdf",
     'threshold': 4,
     'value': 3},
]

"""
This code gives preference to the highest value when the masks have
intersections.
"""


for mask_dict in masks_list:
    label, hdf_file_path, threshold, value = mask_dict['label'], \
                                             mask_dict['file'], \
                                             mask_dict['threshold'], \
                                             mask_dict['value']
    data = _load_hdf_dataset(hdf_file_path=hdf_file_path)
    print("For ", label, "the value is", value)
    output_dataset = 1 * (data >= threshold)
    hdf_mask_name = label + ".hdf"
    output_file_path = join(output_path, hdf_mask_name)
    write_dataset_hdf(output_path=output_file_path, tomo_data=output_dataset)



