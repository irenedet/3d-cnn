from src.python.filereaders.datasets import load_dataset
# import numpy as np
import pandas as pd

path_to_dataset_table = "/struct/mahamid/Irene/yeast/npc/npc_yeast_data.csv"

TOMOS = [
    "180413/006",
    "180413/007",
    "180426/005",
    "180426/006",
    "180426/008",
    "180426/014",
    "180426/027",
    "180426/037",
    "180426/038",
    "180426/040",
    "180426/043",
    "180711/005",
    "180711/006",
    "180711/007",
    "180711/012",
    "180711/017",
    "180711/022",
    "180713/002",
    "180713/005",
    "180713/007",
    "180713/015",
    "180713/018",
    "180713/020",
    "180713/025",
    "180713/027",
    "180713/031",
    "180713/035",
    "180713/037",
    "180713/039",
    "180713/041",
    "180713/043",
    "180713/050",
 ]

for tomo_name in TOMOS:
    print("Tomo", tomo_name)
    df = pd.read_csv(path_to_dataset_table)
    tomo_df = df.loc[df['tomo_name'] == tomo_name].iloc[0]
    path_to_raw = tomo_df['eman2_filetered_tomo']
    z_dim, y_dim, x_dim = load_dataset(path_to_raw).shape
    df.loc[df['tomo_name'] == tomo_name, 'x_dim'] = x_dim
    df.loc[df['tomo_name'] == tomo_name, 'y_dim'] = y_dim
    df.loc[df['tomo_name'] == tomo_name, 'z_dim'] = z_dim
    df.to_csv(path_or_buf=path_to_dataset_table, index=False)


#
# tomos_df = df.loc[df['tomo_name'].isin(TOMOS)]
# for tomo_name in TOMOS:
#     tomo_df = tomos_df[tomos_df['tomo_name'] == tomo_name]
#     path_to_raw = tomo_df.iloc[0]['eman2_filetered_tomo']
#
#     z_dim, y_dim, x_dim = load_dataset(path_to_raw).shape
#     df.loc[df['tomo_name'] == tomo_name, 'x_dim'] = x_dim
#     df.loc[df['tomo_name'] == tomo_name, 'y_dim'] = y_dim
#     df.loc[df['tomo_name'] == tomo_name, 'z_dim'] = z_dim
#
# df.to_csv(path_or_buf=path_to_dataset_table)
