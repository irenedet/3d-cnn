from src.python.filereaders.txt import read_shrec_motl
from src.python.filewriters.h5 import write_hdf_particles_from_motl

path_to_motl = "/struct/mahamid/Processing/shrec_2019_cryoet_dataset/0/particle_locations_model_0.txt"
motl = read_shrec_motl(path_to_motl)

print(motl[0])

output_shape = (200, 512, 512)
hdf_output_path = "/scratch/trueba/shrec/0_test/ALL_particles.hdf"
particle_classes = [n+1 for n in range(12)]

write_hdf_particles_from_motl(path_to_motl,
                              hdf_output_path,
                              output_shape,
                              sphere_radius=8,
                              values_in_motl=False,
                              number_of_particles=None,
                              z_shift=0,
                              particle_classes=particle_classes)
