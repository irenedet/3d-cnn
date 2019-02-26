import csv
import numpy as np

particle_dict = {'4b4t': 1,
                 '2cg9': 2,
                 '3d2f': 3,
                 '3h84': 4,
                 '3cf3': 5,
                 '1u6g': 6,
                 '3gl1': 7,
                 '4d8q': 8,
                 '1s3x': 9,
                 '1qvr': 10,
                 '1bxn': 11,
                 '3qm1': 12}


def read_shrec_motl(path_to_motl: str):
    """
    Output: array whose first entries are the rows of the motif list
    Usage example:
    motl = read_motl_from_csv(path_to_csv_motl)
    list_of_max=[row[0] for row in motl]
    """
    motl = []

    with open(path_to_motl, 'r') as csvfile:
        motlreader = csv.reader(csvfile, delimiter='|')
        for row in motlreader:
            line = row[0].split(" ")
            motl_line = [val for val in line if val != '']
            motl_line = [particle_dict[motl_line[0]]] + \
                        [float(val) for val in motl_line[1:]]
            motl += [motl_line]
    return np.array(motl)
