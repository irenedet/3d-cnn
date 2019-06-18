import numpy as np
from src.python.coordinates_toolbox.utils import \
    extract_coordinates_and_values_from_em_motl
from src.python.filereaders.em import read_em
from src.python.filereaders.csv import read_motl_from_csv
from src.python.filewriters.csv import motl_writer
import os
from src.python.peak_toolbox.utils import \
    extract_motl_coordinates_and_score_values
from os import makedirs

PATHS=[
    "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/181119/002/TM/motl_ED_FAS_1b.em",
    "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/181119/030/TM/motl_ED_FAS_1b.em",
    "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/181126/002/TM/motl_ED_FAS_1b.em",
    "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/181126/012/motl_ED_FAS_1b.em",
    "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/181126/025/TM/motl_ED_FAS_1b.em",
]

NEW_PATHS=[
    "/struct/mahamid/Irene/yeast/ED/181119_002/motl/",
    "/struct/mahamid/Irene/yeast/ED/181119_030/motl/",
    "/struct/mahamid/Irene/yeast/ED/181126_002/motl/",
    "/struct/mahamid/Irene/yeast/ED/181126_012/motl/",
    "/struct/mahamid/Irene/yeast/ED/181126_025/motl/",
]

for path_to_motl, path_to_new_motl in zip(PATHS, NEW_PATHS):
    # path_to_motl = "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/190301/005/TM/motl_ED_FAS_1b.em"
    # path_to_new_motl = "/struct/mahamid/Irene/yeast/ED/190301_005/motl/"
    makedirs(name=path_to_new_motl, exist_ok=True)

    _, motl_extension = os.path.splitext(path_to_motl)
    assert motl_extension == ".em" or motl_extension == ".csv"
    if motl_extension == ".em":
        print("motl clean in .em format")
        Header, motl = read_em(path_to_emfile=path_to_motl)
        values, coordinates = extract_coordinates_and_values_from_em_motl(motl)
    else:
        print("motl clean in .csv format")
        motl = read_motl_from_csv(path_to_motl)
        values, coordinates = extract_motl_coordinates_and_score_values(
            motl)
        coordinates = np.array(coordinates)

    print("value_max=", np.max(values))
    values = np.ones(values.shape)
    print("value_max=", np.max(values))

    bin = 2
    factor = 2 ** (-1 * bin) # bin4 means factor = 4^-1

    binned_coordinates = [factor * np.array(point) for point in coordinates]

    motl_writer(path_to_output_folder=path_to_new_motl, list_of_peak_scores=values,
                list_of_peak_coords=binned_coordinates,in_tom_format=True,
                order_by_score=True, motl_name="motl_ED_FAS_4b.csv")

# path_to_motl_clean = '/scratch/trueba/3d-cnn/clean/180426_005/motl_clean_4b.em'
#
# Extract coordinates from template matching
# Header, motl_clean = read_em(path_to_emfile=path_to_motl_clean)
# coordinates = extract_coordinates_from_em_motl(motl_clean)
# coordinates[:, 0] += np.ones(coordinates.shape[0])
# print(coordinates.shape)
#
# for i in range(3):
#     max = np.max([point[i] for point in coordinates])
#     print(i, max)
