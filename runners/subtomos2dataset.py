from os.path import join
# import re
# import numpy as np
# import h5py
# from src.python.coordinates_toolbox import subtomos
from src.python.filewriters.h5 import write_dataset_from_subtomograms
from src.python.naming import h5_internal_paths

# "/home/papalotl/courses/machine-learning-course-material-2018/exercise_2/Main/pytorch3D"
# data_file = "subtomo_data_path.h5"  # ""tomo004_in_subtomos_128side.h5"
data_dir = "/scratch/trueba/3d-cnn/TEST/"
data_file = "004_in_subtomos_128side.h5"
subtomos_path = join(data_dir, data_file)

output_dir = "/g/scb2/zaugg/trueba/3d-cnn/TEST"
output_file = "merge_subtomos.hdf"
output_path = join(output_dir, output_file)

output_shape = (221, 928, 928)
subtomo_shape = (128, 128, 128)

label_name = "ribosomes"
subtomos_internal_path = join(
    h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS,
    label_name)

# Todo Change the functions in cnn_subtomo_segmentation.py to:
# subtomos_internal_paths = "volumes/predictions/ribosomes"
write_dataset_from_subtomograms(output_path, subtomos_path, output_shape,
                                subtomo_shape, subtomos_internal_path)
