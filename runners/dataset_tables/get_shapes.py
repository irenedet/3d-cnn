from src.python.filereaders.datasets import load_dataset
# import numpy as np
import pandas as pd

path_to_dataset_table = "/struct/mahamid/Irene/yeast/yeast_table.csv"
#     "190301/018",
#     "190301/019",
TOMOS = [
    # "190301/003",
    # "190301/005",
    # "190301/009",
    # "190301/012",
    # "190301/016",
    # "190301/022",
    # "190301/028",
    # "190301/031",
    # "190301/032",
    # "190301/033",
    # "190301/035",
    # "190301/037",
    # "190301/043",
    # "190301/045",
    # "190301/001",
    # "190301/002",
    # "190301/004",
    # "190301/006",
    # "190301/007",
    # "190301/010",
    # "190301/011",
    # "190301/013",
    # "190301/014",
    # "190301/015",
    # "190301/017",
    "190301/020",
    "190301/021",
    "190301/026",
    "190301/029",
    "190301/030",
    "190329/001",
    "190329/004",
    "190329/005",
    "190329/007",
    "190329/010",
    "190329/012",
    "190329/013",
    "190329/015",
    "190329/017",
    "190329/021",
    "190329/022",
    "190329/023",
    "190329/025",
    "190329/028",
    "190329/032",
    "190329/036",
]

for tomo_name in TOMOS:
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
