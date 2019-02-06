import csv
import numpy as np


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
            motl_line = [float(val) for val in line]
            motl += [motl_line]
    return np.array(motl)
