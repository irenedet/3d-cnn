from os.path import join
from src.python.peak_toolbox.subtomos import \
    get_peaks_from_subtomograms_with_overlap
from src.python.filewriters import csv

data_dir = "/scratch/trueba/3d-cnn/TEST/"
data_file = "004_in_subtomos_128side_with_overlap.h5"
subtomos_path = join(data_dir, data_file)

list_of_maxima, list_of_maxima_coords = \
    get_peaks_from_subtomograms_with_overlap(
        subtomo_file_path=subtomos_path, label_name="ribosomes",
        subtomo_shape=(128, 128, 128), output_shape=(221, 928, 928),
        min_peak_distance=12,
        overlap=12)

csv.motl_writer(path_to_output_folder=data_dir,
               list_of_peak_scores=list_of_maxima,
               list_of_peak_coords=list_of_maxima_coords)

print("Finished!")