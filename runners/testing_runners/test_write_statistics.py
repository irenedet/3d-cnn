import os

import pandas as pd


# import numpy as np


def write_statistics(statistics_file: str, statistics_label: str,
                     tomo_name: str, stat_measure: float):
    dict_stats = {'tomo_name': [tomo_name],
                  statistics_label: [stat_measure]}
    mini_stats_df = pd.DataFrame(dict_stats)
    if os.path.isfile(statistics_file):
        print("The statistics file exists")
        stats_df = pd.read_csv(statistics_file)
        stats_df['tomo_name'] = stats_df['tomo_name'].astype(str)
        stats_df = stats_df.append(mini_stats_df, sort=False)
        stats_df.to_csv(path_or_buf=statistics_file, index=False)

    else:
        print("The statistics file does not exist, we will create it.")
        mini_stats_df.to_csv(path_or_buf=statistics_file, index=False)
    return


# statistics_label = 'NO_DA_sigmoid_70S_50S_memb_D_2_IF_8_pr_radius_20'
# tomo_name = ""
# auPRC = 0

# write_statistics(statistics_file=statistics_file,
#                  statistics_label=statistics_label,
#                  tomo_name=tomo_name,
#                  stat_measure=auPRC)

statistics_file = "/struct/mahamid/Irene/liang_data/multiclass/peak_statistics_class_0.csv"

stats_df = pd.read_csv(statistics_file)
print(stats_df)
