import re
import numpy as np


def get_coord_from_name(subtomo_name):
    return [int(val) for val in re.findall(r'\d+', subtomo_name)]


def get_subtomo_corners(output_shape: tuple, subtomo_shape: tuple,
                        subtomo_center: tuple) -> tuple:
    subtomo_l1radius = subtomo_shape[0] // 2, subtomo_shape[1] // 2, \
                       subtomo_shape[2] // 2
    init_points = [center_dim - subtomo_dim for center_dim, subtomo_dim
                   in zip(subtomo_center, subtomo_l1radius)]
    end_points = [center_dim + subtomo_dim for center_dim, subtomo_dim
                  in zip(subtomo_center, subtomo_l1radius)]
    end_points = [np.min((end_point, tomo_dim)) for end_point, tomo_dim
                  in zip(end_points,
                         output_shape)]
    lengths = [end - init for end, init in zip(end_points, init_points)]
    return init_points, end_points, lengths
