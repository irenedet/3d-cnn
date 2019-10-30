from src.python.filereaders.datasets import load_dataset
# import numpy as np
import pandas as pd

# path_to_dataset_table = "/struct/mahamid/Irene/yeast/npc/npc_yeast_data.csv"

TOMOS = [
    "190218/043",
    "190218/044",
    "190218/048",
    "190218/049",
    "190218/050",
    "190218/051",
    "190218/052",
    "190218/054",
    "190218/056",
    "190218/059",
    "190218/060",
    "190218/061",
    "190218/062",
    "190218/063",
    "190218/064",
    "190218/065",
    "190218/066",
    "190218/067",
    "190218/068",
    "190218/069",
    "190218/070",
    "190218/071",
    "190218/072",
    "190218/073",
    "190218/075",
    "190218/076",
    "190218/077",
    "190218/078",
    "190218/081",
    "190218/082",
    "190218/083",
    "190218/084",
    "190218/085",
    "190218/086",
    "190218/087",
    "190218/088",
    "190218/089",
    "190218/090",
    "190218/091",
    "190218/092",
    "190218/093",
    "190218/094",
    "190218/095",
    "190218/096",
    "190218/097",
    "190218/098",
    "190218/099",
    "190218/100",
    "190218/101",
    "190218/102",
    "190218/103",
    "190218/104",
    "190218/105",
    "190218/106",
    "190218/108",
    "190218/110",
    "190218/111",
    "190218/112",
    "190218/113",
    "190218/114",
    "190218/115",
    "190218/116",
    "190218/117",
    "190218/118",
    "190218/119",
    "190218/120",
    "190218/121",
    "190218/122",
    "190218/123",
    "190218/124",
    "190218/125",
    "190223/129",
    "190223/130",
    "190223/131",
    "190223/132",
    "190223/133",
    "190223/134",
    "190223/135",
    "190223/136",
    "190223/139",
    "190223/140",
    "190223/141",
    "190223/142",
    "190223/143",
    "190223/144",
    "190223/145",
    "190223/146",
    "190223/148",
    "190223/149",
    "190223/151",
    "190223/152",
    "190223/153",
    "190223/154",
    "190223/155",
    "190223/156",
    "190223/157",
    "190223/159",
    "190223/160",
    "190223/162",
    "190223/163",
    "190223/165",
    "190223/166",
    "190223/168",
    "190223/169",
    "190223/171",
    "190223/172",
    "190223/173",
    "190223/174",
    "190223/175",
    "190223/176",
    "190223/177",
    "190223/178",
    "190223/179",
    "190223/180",
    "190223/181",
    "190223/182",
    "190223/183",
    "190223/184",
    "190223/185",
    "190223/186",
    "190223/187",
    "190223/188",
    "190223/189",
    "190223/190",
    "190223/191",
    "190223/192",
    "190223/193",
    "190223/194",
]

for tomo_name in TOMOS:
    print("Tomo", tomo_name)
    path_to_raw = "/struct/mahamid/Irene/NPC/SPombe/" + tomo_name +\
                  "/double_eman_filtered_raw_4b.hdf"
    dataset = load_dataset(path_to_dataset=path_to_raw)
    print(dataset.shape)
    # df = pd.read_csv(path_to_dataset_table)
    # tomo_df = df.loc[df['tomo_name'] == tomo_name].iloc[0]
    # path_to_raw = tomo_df['eman2_filetered_tomo']
    # z_dim, y_dim, x_dim = load_dataset(path_to_raw).shape
    # df.loc[df['tomo_name'] == tomo_name, 'x_dim'] = x_dim
    # df.loc[df['tomo_name'] == tomo_name, 'y_dim'] = y_dim
    # df.loc[df['tomo_name'] == tomo_name, 'z_dim'] = z_dim
    # df.to_csv(path_or_buf=path_to_dataset_table, index=False)

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
