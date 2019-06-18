from os.path import join
from src.python.filewriters.csv import \
    write_global_motl_from_overlapping_subtomograms

data_dir = "/scratch/trueba/3d-cnn/TEST/"
data_file = "004_in_subtomos_128side_with_overlap.h5"
subtomos_path = join(data_dir, data_file)
motl_output_dir = "/scratch/trueba/3d-cnn/TEST/motl_unique"
label_name = "ribosomes"
subtomo_shape = (128, 128, 128)
output_shape = (221, 927, 927)
numb_peaks = 50
min_peak_distance = 12
overlap = 12
number_peaks_uniquify = 7000
write_global_motl_from_overlapping_subtomograms(subtomos_path,
                                                motl_output_dir,
                                                overlap,
                                                label_name,
                                                output_shape,
                                                subtomo_shape,
                                                numb_peaks,
                                                min_peak_distance,
                                                number_peaks_uniquify)
