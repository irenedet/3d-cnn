from os.path import join
from src.python.filereaders.hdf import _load_hdf_dataset
from src.python.filewriters.h5 import write_dataset_hdf
from os import makedirs
import numpy as np

data_files = [
    "/struct/mahamid/Irene/yeast/vpp/180426_004/memb/binarized_mask.hdf",
    "/struct/mahamid/Irene/yeast/vpp/180426_004/004_bin4.hdf"
]

output_dirs =[
    "/struct/mahamid/Irene/yeast/vpp/180426_004/memb/binarized_mask_sq.hdf",
    "/struct/mahamid/Irene/yeast/vpp/180426_004/004_bin4_sq.hdf"
]


# output_dir = "/struct/mahamid/Irene/yeast/ribosomes/180426_004/fas_corrected/"
# threshold = 0  # fas: 0.296
# value = 1
x_shifts = [0, 0]
dataset_shape = (1000, 927, 927)
dimz, dimy, dimx = dataset_shape
# output_dataset = np.zeros(dataset_shape)
for output_file_path, hdf_file_path in zip(output_dirs, data_files):
    # makedirs(name=output_file_path, exist_ok=True)
    tomo_data = _load_hdf_dataset(hdf_file_path=hdf_file_path)
    tomo_data = np.array(tomo_data)
    # minimum_shape = [min([dim1, dim2]) for dim1, dim2 in zip(dataset_shape, tomo_data.shape)]
    # nz, ny, nx = minimum_shape
    # print(minimum_shape)
    # output_dataset[:nz, :ny, :nx] += tomo_data[:nz, :ny, :nx]
    output_dataset = tomo_data[:dimz, :dimy, :dimx]
    write_dataset_hdf(output_path=output_file_path, tomo_data=output_dataset)

# A = [1,2,3,4,5,6]
# x = 0
# print(len(A[:6-x]))
# print(len(A[x:6]))
