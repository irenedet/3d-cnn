output_path = "/scratch/trueba/3d-cnn/clean/180426_004/full_dataset.hdf"
path_to_hdf = "/scratch/trueba/3d-cnn/clean/180426_004/subtomo380-600.hdf"

z_shift = 380  # shift between original tomogram and subtomogram of analysis
y_shift = 0
x_shift = 0
# output
shape_x = 959
shape_y = 927
shape_z = 1000
radius = 10

coords_in_tom_format = 'True'

from src.python.filereaders.hdf import _load_hdf_dataset
from src.python.filewriters.h5 import write_dataset_hdf
import numpy as np

mask = _load_hdf_dataset(hdf_file_path=path_to_hdf)
mask = np.array(mask)
new_shape = (shape_z, shape_y, shape_x)
shift_vector = (z_shift, y_shift, x_shift)

new_mask = np.zeros(new_shape)

print(new_mask.shape, mask.shape)

min_shape = [np.min([mask_sh, dim_shift + old_sh]) for
             mask_sh, dim_shift, old_sh
             in zip(new_shape, shift_vector, mask.shape)]
min_shape_z, min_shape_y, min_shape_x = min_shape
print(min_shape)
new_mask[z_shift:min_shape_z, y_shift:min_shape_y, x_shift:min_shape_x] = \
    mask[:min_shape_z - z_shift, :min_shape_y - y_shift, :min_shape_x - x_shift]

write_dataset_hdf(output_path=output_path, tomo_data=new_mask)
