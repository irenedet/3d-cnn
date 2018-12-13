from os.path import join
import re
import numpy as np
import h5py


def get_corner_coord(subtomo_name):
    return [int(val) for val in re.findall(r'\d+', subtomo_name)]


def write_dataset_h5(output_path: str, tomo_data: np.array):
    with h5py.File(output_path, 'w') as f:
        h5_internal_path = "MDF/images/0/image"
        f[h5_internal_path] = tomo_data


def get_cropping_points(output_shape: tuple, subtomo_shape: tuple,
                        subtomo_center: tuple) -> tuple:
    subtomo_l1radius = subtomo_shape[0] // 2, subtomo_shape[1] // 2, \
                       subtomo_shape[2] // 2
    print(subtomo_l1radius)
    print(subtomo_center)
    init_points = [center_dim - subtomo_dim for center_dim, subtomo_dim
                   in zip(subtomo_center, subtomo_l1radius)]
    end_points = [center_dim + subtomo_dim for center_dim, subtomo_dim
                  in zip(subtomo_center, subtomo_l1radius)]
    end_points = [np.min((end_point, tomo_dim)) for end_point, tomo_dim
                  in zip(end_points,
                         output_shape)]
    lengths = [end - init for end, init in zip(end_points, init_points)]
    return init_points, end_points, lengths


def write_dataset_from_subtomograms(output_path, subtomo_path, output_shape,
                                    subtomo_shape):
    tomo_data = np.zeros(output_shape)
    with h5py.File(subtomo_path, 'r') as f:
        for subtomo_name in list(f['volumes/labels/ribosomes']):
            subtomo_center = get_corner_coord(subtomo_name)
            init_points, end_points, lengths = get_cropping_points(output_shape,
                                                                   subtomo_shape,
                                                                   subtomo_center)
            print(init_points, end_points, lengths)
            subtomo_h5_internal_path = join('volumes/labels/ribosomes',
                                            subtomo_name)
            # A = f[subtomo_h5_internal_path][0, 0:lengths[0], 0:lengths[1],
            #     0:lengths[2]]
            # print("shape of A", A.shape)
            tomo_data[init_points[0]: end_points[0],
            init_points[1]: end_points[1], init_points[2]: end_points[2]] = \
                f[subtomo_h5_internal_path][0, 0:lengths[0], 0:lengths[1],
                0:lengths[2]]
    write_dataset_h5(output_path, tomo_data)
    print("right before deleting", np.max(tomo_data))
    del tomo_data


folder_path = "/home/papalotl/courses/machine-learning-course-material-2018/exercise_2/Main/pytorch3D"
# subtomo_path = join(folder_path, "subtomo_data_path.h5")
data_dir = "/scratch/trueba/3d-cnn/evaluating_data/"
data_file = "subtomo_data_path.h5"  # ""tomo004_in_subtomos_128side.h5"
subtomo_path = join(data_dir, data_file)
output_path = join(folder_path, "test_merge_subtomos.hdf")

# TODO modify to accept non 2^n shapes
output_shape = (221, 928, 928)
# output_shape = (256, 1024, 1024)
# output_shape = (128, 128, 128)
subtomo_shape = (128, 128, 128)

write_dataset_from_subtomograms(output_path, subtomo_path, output_shape,
                                subtomo_shape)
