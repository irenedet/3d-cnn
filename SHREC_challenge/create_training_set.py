from os.path import join
from os import makedirs

from src.python.filereaders.hdf import _load_hdf_dataset

path_to_raw = "/scratch/trueba/shrec/0_test/central_region.hdf"
output_dir = "/scratch/trueba/shrec/0_test/"
makedirs(name=output_dir, exist_ok=True)
hdf_output_path = join(output_dir, "particles1.hdf")

path_to_labeled = hdf_output_path

output_h5_file_name = "particles_training.h5"
output_h5_file_path = join(output_dir, output_h5_file_name)

output_shape = (200, 512, 512)
subtomogram_shape = (128, 128, 128)
overlap = 12

label_name = "particle1"

raw_dataset = _load_hdf_dataset(hdf_file_path=path_to_raw)
labels_dataset = _load_hdf_dataset(hdf_file_path=path_to_labeled)
from src.python.datasets.actions import partition_raw_and_labels_tomograms

partition_raw_and_labels_tomograms(raw_dataset=raw_dataset,
                                   labels_dataset=labels_dataset,
                                   label_name=label_name,
                                   output_h5_file_path=output_h5_file_path,
                                   subtomo_shape=subtomogram_shape,
                                   overlap=overlap)

print("The script has finished!")
