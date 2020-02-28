from file_actions.readers.shrec import read_shrec_motl
from file_actions.writers.h5 import write_hdf_particles_from_motl

path_to_motl = "/struct/mahamid/Processing/shrec_2019_cryoet_dataset/0/particle_locations_model_0.txt"
motl = read_shrec_motl(path_to_motl)

print(motl[0])

output_shape = (512, 512, 512)
hdf_output_path = "/scratch/trueba/shrec/0_sph_masks/top5_particles_differentiated.hdf"
# particle_classes = ['3qm1', '1s3x', '3h84', '3gl1', '2cg9', '3d2f', '1u6g',
#                     '3cf3', '1bxn', '1qvr', '4b4t', '4d8q']
particle_classes = ['3cf3', '1bxn', '1qvr', '4b4t', '4d8q']

write_hdf_particles_from_motl(path_to_motl,
                              hdf_output_path,
                              output_shape,
                              values_in_motl=True,
                              z_shift=156,
                              particle_classes=particle_classes)
