import pandas as pd

from filereaders.datasets import load_dataset

# motive_list_1 = "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/018/TM/motl_clean_4b.em"
# motive_list_2 = "/struct/mahamid/Shared/For_Irene/predictions/180711/018/motl_459_checked.csv"
motive_list_1 = "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/004/TM/motl_clean_4b.em"
motive_list_2 = "/struct/mahamid/Shared/For_Irene/predictions/180711/004/motl_587_checked.csv"

output_motl_path = "/struct/mahamid/Irene/yeast/healthy/180711/004/ribos/motl/corrected_motl_verified.csv"

motl_1 = load_dataset(motive_list_1)
motl_1 = pd.DataFrame(motl_1)
motl_2 = pd.read_csv(motive_list_2, header=None)
print(motl_1.shape)
print(motl_2.shape)
combined_motl = motl_1.append(motl_2, sort=False)
print(combined_motl)
combined_motl.to_csv(output_motl_path, header=False, index=False)
