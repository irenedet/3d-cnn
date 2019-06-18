from os.path import join
from src.python.filewriters.h5 import write_hdf_particles_from_motl


# ToDo check correctness of the code here!
data_dir = "/scratch/trueba/3d-cnn/TEST/"
data_file = "004_from_motl_4444.hdf"
hdf_output_path = join(data_dir, data_file)
output_shape = (221, 928, 928)
path_to_csv_motl = "/scratch/trueba/3d-cnn/TEST/motl_unique/motl_4444.csv"

write_hdf_particles_from_motl(path_to_motl=path_to_csv_motl,
                              hdf_output_path=hdf_output_path,
                              output_shape=output_shape,
                              sphere_radius=8,
                              values_in_motl=True,
                              number_of_particles=1000)
