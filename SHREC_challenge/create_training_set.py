from os.path import join
from os import makedirs

from src.python.filereaders.hdf import _load_hdf_dataset
from src.python.datasets.transformations import transform_data_from_h5

path_to_raw = "/scratch/trueba/shrec/0/reconstruction_model_0.hdf"
# path_to_labeled = "/struct/mahamid/Irene/shrec/0/all_classes_differentiated_mask_0.hdf"
path_to_labeled = "/struct/mahamid/Irene/shrec/0/all_classes_foreground_mask_0.hdf"

output_dir = "/scratch/trueba/shrec/0_real_masks/training_sets/"
makedirs(name=output_dir, exist_ok=True)
output_h5_file_name = "all_particles_foreground_training.h5"
output_h5_file_path = join(output_dir, output_h5_file_name)

output_shape = (512, 512, 512)
subtomogram_shape = (64, 64, 64)
overlap = 12

label_name = "all_particles"

raw_dataset = _load_hdf_dataset(hdf_file_path=path_to_raw)
labels_dataset = _load_hdf_dataset(hdf_file_path=path_to_labeled)
from src.python.datasets.actions import partition_raw_and_labels_tomograms

partition_raw_and_labels_tomograms(raw_dataset=raw_dataset,
                                   labels_dataset=labels_dataset,
                                   label_name=label_name,
                                   output_h5_file_path=output_h5_file_path,
                                   subtomo_shape=subtomogram_shape,
                                   overlap=overlap)

