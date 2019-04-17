from os.path import join
from os import makedirs
from src.python.filewriters.h5 import \
    write_dataset_from_subtomos_with_overlap_multiclass
from src.python.naming import h5_internal_paths

subtomos_path = "/scratch/trueba/3d-cnn/training_data/dice-multi-class/004/G_sigma1/train_and_test_partitions/partition_training.h5"

output_dir = "/scratch/trueba/3d-cnn/cnn_evaluation/multi-class/004/G_sigma1_D5_IF8"
makedirs(name=output_dir, exist_ok=True)

output_shape = (221, 928, 928)
subtomo_shape = (128, 128, 128)

overlap_thickness = 12
output_classes = 3
label_name = "D_2_IF_8_w_1_1_1"
segmentation_names = ['ribos', 'fas', 'memb'] #in the original file
subtomos_internal_path = join(
    h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS,
    label_name)
for n in range(output_classes):
    segmentation_name = segmentation_names[n]
    output_file = segmentation_name + ".hdf"
    output_path = join(output_dir, output_file)
    class_number = n
    write_dataset_from_subtomos_with_overlap_multiclass(
        output_path,
        subtomos_path,
        output_shape,
        subtomo_shape,
        subtomos_internal_path,
        class_number,
        overlap_thickness)
