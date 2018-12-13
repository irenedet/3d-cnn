import numpy as np


def crop_tensor(input: np.array, shape_to_crop: tuple) -> np.array:
    """
    Function from A. Kreshuk to crop tensors of order 3, starting always from
    the origin.
    :param input: the input np.array image
    :param shape_to_crop: a tuple (cz, cy, cx), where each entry corresponds
    to the size of the  cropped region along each axis.
    :return: np.array of size (cz, cy, cx)
    """
    input_shape = input.shape
    assert all(ish >= csh for ish, csh in zip(input_shape, shape_to_crop)), \
        "Input shape must be larger equal crop shape"
    # get the difference between the shapes
    shape_diff = tuple((ish - csh) // 2
                       for ish, csh in zip(input_shape, shape_to_crop))
    # calculate the crop
    crop = tuple(slice(sd, sh - sd)
                 for sd, sh in zip(shape_diff, input_shape))
    return input[crop]


def crop_window(input, shape_to_crop, window_corner):
    """
    Function from A. Kreshuk to crop tensors of order 3, starting always
    from a given corner.
    :param input: the input np.array image
    :param shape_to_crop: a tuple (cz, cy, cx), where each entry corresponds
    to the size of the  cropped region along each axis.
    :param window_corner: point from where the window will be cropped.
    :return: np.array of size (cz, cy, cx)
    """
    input_shape = input.shape
    assert all(ish >= csh for ish, csh in zip(input_shape, shape_to_crop)), \
        "Input shape must be larger equal crop shape"
    # get the difference between the shapes
    crop = tuple(slice(wc, wc + csh)
                 for wc, csh in zip(window_corner, shape_to_crop))
    # print(crop)
    return input[crop]


def crop_window_around_point(input, shape_to_crop_zyx, window_center_zyx):
    input_shape = input.shape
    assert all(ish - csh // 2 - center >= 0 for ish, csh, center in
               zip(input_shape, shape_to_crop_zyx, window_center_zyx)), \
        "Input shape must be larger or equal than crop shape"
    assert all(center - csh // 2 >= 0 for csh, center in
               zip(shape_to_crop_zyx, window_center_zyx)), \
        "Input shape around window center must be larger equal than crop shape"
    # get the difference between the shapes
    crop = tuple(slice(center - csh // 2, center + csh // 2)
                 for csh, center in zip(shape_to_crop_zyx, window_center_zyx))
    return input[crop]
