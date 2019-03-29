from os.path import join
from src.python.filewriters.h5 import \
    write_dataset_from_subtomos_with_overlap, \
    write_dataset_from_subtomos_with_overlap_multiclass, \
    write_dataset_from_subtomos_with_overlap_multiclass_exponentiating
from src.python.naming import h5_internal_paths

subtomos_path = "/scratch/trueba/shrec/0_sph_masks/training_sets/all_foreground_training.h5"

output_dir = "/scratch/trueba/shrec/0_sph_masks/cnn_evaluations/all_fore_D2_IF8"
output_file = "all_foreground.hdf"
output_path = join(output_dir, output_file)

output_shape = (512, 512, 512)
subtomo_shape = (64, 64, 64)

overlap_thickness = 12

label_name = "all_particles"
# data_dir = "/scratch/trueba/3d-cnn/TEST/"
# data_file = "004_in_subtomos_128side_with_overlap.h5"
# subtomos_path = join(data_dir, data_file)
#
# output_dir = "/g/scb2/zaugg/trueba/3d-cnn/TEST"
# output_file = "merge_subtomos_with_overlap.hdf"
# output_path = join(output_dir, output_file)
#
# output_shape = (221, 928, 928)
# subtomo_shape = (128, 128, 128)
#
# overlap_thickness = 12
#
# label_name = "ribosomes"
subtomos_internal_path = join(
    h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS,
    label_name)

write_dataset_from_subtomos_with_overlap_multiclass(
    output_path,
    subtomos_path,
    output_shape,
    subtomo_shape,
    subtomos_internal_path,
    overlap_thickness)
