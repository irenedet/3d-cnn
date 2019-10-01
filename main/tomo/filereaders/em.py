import numpy as np
from collections import defaultdict


def _dtypes_mapping() -> defaultdict:
    """
    Tom format data types mapping.
    """
    def default_type():
        print("dtype was undefined, by default it wil be set to np.double")
        return np.double

    dtypes_default = defaultdict(default_type)
    dtypes = {
        1: np.byte,
        2: np.int16,
        4: np.int32,
        5: np.float32,
        8: np.complex64,
        9: np.double
    }
    dtypes_default.update(dtypes)
    return dtypes_default


def _reshape_if_2d(value: np.array):
    """
    Checks if a dataset is 2d. If so, converts to 2d.
    """
    if value.shape[0] == 1 and len(value.shape) == 3:
        value = value[0, :, :]
    return value


def read_em(path_to_emfile: str) -> tuple:
    """
    Reads an .em dataset (in tom format).
    where header is a dictionary specifying
     - 'Machine_Coding'
     - 'version'
     - 'old_param'
     - 'data_type_code'
     - 'image_dimensions'
     - 'the_rest'
    The value is an array.
    For more information check tom_emread.
    :param path_to_emfile: str, pointing to the .em file
    :return: (header, value)
    """
    with open(path_to_emfile, 'r') as f:
        header = dict()
        header['Machine_Coding'] = np.fromfile(f, dtype=np.byte, count=1)
        header['version'] = np.fromfile(f, dtype=np.byte, count=1)
        header['old_param'] = np.fromfile(f, dtype=np.byte, count=1)
        header['data_type_code'] = np.fromfile(f, dtype=np.byte, count=1)
        header['image_dimensions'] = np.fromfile(f, dtype=np.int32, count=3)
        header['the_rest'] = np.fromfile(f, dtype=np.byte, count=496)
        dtypes_mapping = _dtypes_mapping()
        dtype = dtypes_mapping[header['data_type_code']]
        new_image_dim = header['image_dimensions'][::-1]
        header['image_dimensions'] = np.array(new_image_dim)
        value = np.fromfile(f, dtype=dtype)
        value = np.reshape(value, header['image_dimensions'])
        value = _reshape_if_2d(value)
    return header, value
