import csv
import numpy as np
import pandas as pd


def read_motl_from_csv(path_to_csv_motl: str):
    """
    Output: array whose first entries are the rows of the motif list
    Usage example:
    motl = read_motl_from_csv(path_to_csv_motl)
    list_of_max=[row[0] for row in motl]
    """
    motl = []

    with open(path_to_csv_motl, 'r') as csvfile:
        motlreader = csv.reader(csvfile, delimiter='|')
        for row in motlreader:
            line = row[0].split(",")
            motl_line = [float(val) for val in line[:20]]
            motl += [motl_line]
    return np.array(motl)


def load_motl_as_df(path_to_motl):
    column_names = ['score', 'x_', 'y_', 'peak', 'tilt_x', 'tilt_y', 'tilt_z',
                    'x', 'y', 'z', 'empty_1', 'empty_2', 'empty_3', 'x-shift',
                    'y-shift', 'z-shift', 'phi', 'psi', 'theta', 'class']
    motl_df = pd.read_csv(path_to_motl, header=None)
    motl_df.columns = column_names
    return motl_df
