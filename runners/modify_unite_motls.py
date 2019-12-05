from os import makedirs

from filewriters.csv import unite_motls

min_peak_distance = 16
output_dir = "/struct/mahamid/Irene/yeast/vpp/180426_004/fas/motl"
path_to_motl_extra = "/struct/mahamid/Shared/For_Irene/predictions/180426/004/fas/motl_309_class_0_checked_clean.csv"
path_to_motl_old = "/scratch/trueba/3d-cnn/clean/180426_004/motl_clean_fas_4b_iniavg.em"
motl_name = "corrected_motl.csv"

makedirs(name=output_dir, exist_ok=True)

unite_motls(path_to_motl1=path_to_motl_old,
            path_to_motl2=path_to_motl_extra,
            path_to_output_motl_dir=output_dir,
            output_motl_name=motl_name)
