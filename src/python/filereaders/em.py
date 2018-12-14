import numpy as np


def load_motl(path_to_emfile) -> tuple:
    # Function that reads a em file
    with open(path_to_emfile, 'r') as f:
        header = dict()
        header['Machine_Coding'] = np.fromfile(f, dtype=np.byte, count=1)
        header['version'] = np.fromfile(f, dtype=np.byte, count=1)
        header['old_param'] = np.fromfile(f, dtype=np.byte, count=1)
        header['data_type_code'] = np.fromfile(f, dtype=np.byte, count=1)
        header['image_dimensions'] = np.fromfile(f, dtype=np.int32, count=3)
        header['the_rest'] = np.fromfile(f, dtype=np.byte, count=496)
        if header['data_type_code'] == 1:
            dtype = np.byte
        if header['data_type_code'] == 2:
            dtype = np.int16
        if header['data_type_code'] == 4:
            dtype = np.int32
        if header['data_type_code'] == 5:
            dtype = np.float32
        if header['data_type_code'] == 8:
            dtype = np.complex64
        if header['data_type_code'] == 9:
            dtype = np.double
        new_image_dim = header['image_dimensions'][::-1]
        header['image_dimensions'] = np.array(new_image_dim)
        value = np.fromfile(f, dtype=dtype)
        value = np.reshape(value, header['image_dimensions'])
    return header, value


def _extract_coordinate_and_values(motl):
    header, value = motl
    coordinates = value[0, :, 7:10]  # TODO comment meaning of 7:10
    score_value = value[0, :, 1]  # TODO comment meaning of 7:10
    return coordinates, score_value


def load_coordinate_and_score_values(path_to_emfile) -> tuple:
    motl = load_motl(path_to_emfile)
    return _extract_coordinate_and_values(motl)
