from os.path import join
from os import makedirs
from src.python.filewriters.h5 import \
    write_dataset_from_subtomos_with_overlap_dice_multiclass
from src.python.naming import h5_internal_paths

subtomos_path = "/scratch/trueba/3d-cnn/training_data/dice-multi-class/004/G_sigma1/train_and_test_partitions/partition_training.h5"

output_dir = "/scratch/trueba/3d-cnn/training_data/dice-multi-class/004/clean_masks/"
makedirs(name=output_dir, exist_ok=True)

output_shape = (221, 928, 928)
subtomo_shape = (128, 128, 128)

overlap_thickness = 12
output_classes = 3
# label_name = "D_2_IF_8_w_1_1_1"
segmentation_names = ['ribo', 'fas', 'memb'] #in the original file
# subtomos_internal_path = join(
#     h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS,
#     label_name)

import h5py
path = "/scratch/trueba/3d-cnn/training_data/dice-multi-class/004/G_sigma1/train_and_test_partitions/partition_training.h5"
with h5py.File(path, "r") as f:
    print(list(f[h5_internal_paths.LABELED_SUBTOMOGRAMS]))

subtomos_internal_path = h5_internal_paths.LABELED_SUBTOMOGRAMS

for n in range(output_classes):
    segmentation_name = segmentation_names[n]
    subtomos_internal_path_class = join(subtomos_internal_path,
                                        segmentation_name)
    output_file = segmentation_name + ".hdf"
    output_path = join(output_dir, output_file)
    class_number = n
    write_dataset_from_subtomos_with_overlap_dice_multiclass(
        output_path,
        subtomos_path,
        output_shape,
        subtomo_shape,
        subtomos_internal_path_class,
        class_number,
        overlap_thickness)
