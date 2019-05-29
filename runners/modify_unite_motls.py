from os import makedirs

min_peak_distance = 16
output_dir = "/struct/mahamid/Irene/yeast/vpp/180426_004/fas/motl"
path_to_motl_extra = "/struct/mahamid/Shared/For_Irene/predictions/180426/004/fas/motl_309_class_0_checked_clean.csv"
path_to_motl_old = "/scratch/trueba/3d-cnn/clean/180426_004/motl_clean_fas_4b_iniavg.em"
motl_name = "corrected_motl.csv"

makedirs(name=output_dir, exist_ok=True)

from src.python.filewriters.csv import write_union_of_motls
# Todo this apparently does not work right!!
write_union_of_motls(path_to_motl_1=path_to_motl_old,
                     path_to_motl_2=path_to_motl_extra,
                     path_to_output_folder=output_dir,
                     in_tom_format=True,
                     motl_name=motl_name)
