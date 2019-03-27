from os.path import join
from os import makedirs
from src.python.filewriters.h5 import \
    write_dataset_from_subtomos_with_overlap_multiclass
from src.python.naming import h5_internal_paths

subtomos_path = "/scratch/trueba/shrec/0_real_masks/training_sets/all_particles_foreground_training.h5"

output_dir = "/scratch/trueba/shrec/0_real_masks/evaluation/all_foreground/"
makedirs(name=output_dir, exist_ok=True)


output_shape = (512, 512, 512)
subtomo_shape = (64, 64, 64)

overlap_thickness = 12

label_name = "all_particles"

subtomos_internal_path = join(
    h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS,
    label_name)
for n in range(1):
    output_file = "particle_" + str(n) + ".hdf"
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
