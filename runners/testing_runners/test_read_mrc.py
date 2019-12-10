from filereaders.mrc import read_mrc
from filewriters.h5 import write_dataset_hdf

mrc_files = [
    # "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/181119/002/etomo/bin4/002_sq_df_sorted.rec",
    # "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/181119/030/etomo/bin4/030_sq_df_sorted.rec",
    "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/181126/002/etomo/bin4/002_sq_df_sorted.rec",
    "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/181126/012/etomo/bin4/012_sq_df_sorted.rec",
    "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/181126/025/etomo/bin4/025_sq_df_sorted.rec"
]

output_files = [
    # "/struct/mahamid/Irene/yeast/ED/181119/002/raw_tomo.hdf",
    # "/struct/mahamid/Irene/yeast/ED/181119/030/raw_tomo.hdf",
    "/struct/mahamid/Irene/yeast/ED/181126/002/raw_tomo.hdf",
    "/struct/mahamid/Irene/yeast/ED/181126/012/raw_tomo.hdf",
    "/struct/mahamid/Irene/yeast/ED/181126/025/raw_tomo.hdf",
]
for path_to_mrc, output_path in zip(mrc_files, output_files):
    print("processing", path_to_mrc)
    tomo = read_mrc(path_to_mrc=path_to_mrc)

    print("tomo.shape", tomo.shape)

    write_dataset_hdf(output_path=output_path, tomo_data=tomo)
