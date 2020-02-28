from os.path import join
from os import makedirs
from file_actions.writers.h5 import \
    assemble_tomo_from_subtomos
from src.python.constants import h5_internal_paths


# subtomos_path = "/scratch/trueba/shrec/0_sph_masks/training_sets/all_foreground_training.h5"
#
# output_dir = "/scratch/trueba/shrec/0_sph_masks/cnn_evaluations/all_fore_D2_IF8"
# output_file = "all_foreground.hdf"
# output_path = join(output_dir, output_file)
#
# output_shape = (512, 512, 512)
# subtomo_shape = (64, 64, 64)
#
# overlap_thickness = 12
#
# label_name = "all_particles"

subtomos_path = "/scratch/trueba/shrec/0_sph_masks/training_sets/all_differentiated_training.h5"

output_dir = "/scratch/trueba/shrec/0_sph_masks/cnn_evaluations/all_diff_D3_IF8"
makedirs(name=output_dir, exist_ok=True)


output_shape = (512, 512, 512)
subtomo_shape = (64, 64, 64)

overlap_thickness = 12

label_name = "all_particles"

subtomos_internal_path = join(
    h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS,
    label_name)
for n in range(13):
    output_file = "particle_" + str(n) + ".hdf"
    output_path = join(output_dir, output_file)
    class_number = n
    assemble_tomo_from_subtomos(
        output_path,
        subtomos_path,
        output_shape,
        subtomo_shape,
        subtomos_internal_path,
        class_number,
        overlap_thickness)
