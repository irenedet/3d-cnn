import numpy as np
import scipy.ndimage as scimg


def ZXZ_rotate_ref(ref: np.array, ZXZ_angles_in_degrees: tuple,
                   axis_in_tom_format=False):
    """
        Rotation of a given 3d image from Euler angles given in the ZXZ
    convention.
    :param ref: dataset to which the rotation will be applied
    :param ZXZ_angles_in_degrees: Euler angles (phi, psi, theta), where:
    1. a rot around z of angle phi is applied first
    2. a rot around x of angle psi is applied second
    3. a rot around z of angle theta is applied third
    :param axis_in_tom_format: bool, by default False. If it is set to true,
    it is assumed that the dataset is read in coordinates x, y, z.
    :return: zxz_rotated_ref: a rotated dataset of the same size as the original ref.
    """
    phi, psi, theta = ZXZ_angles_in_degrees

    z_rotated_ref = np.zeros(ref.shape)
    xz_rotated_ref = np.zeros(ref.shape)
    zxz_rotated_ref = np.zeros(ref.shape)

    if axis_in_tom_format:
        scimg.rotate(input=ref, angle=phi, axes=(0, 1), reshape=False,
                     output=z_rotated_ref, order=1, mode='constant', cval=0.0,
                     prefilter=False)
        scimg.rotate(input=z_rotated_ref, angle=psi, axes=(1, 2), reshape=False,
                     output=xz_rotated_ref, order=1, mode='constant', cval=0.0,
                     prefilter=False)
        scimg.rotate(input=xz_rotated_ref, angle=theta, axes=(0, 1),
                     reshape=False, output=zxz_rotated_ref, order=1,
                     mode='constant', cval=0.0, prefilter=False)
    else:
        scimg.rotate(input=ref, angle=phi, axes=(2, 1), reshape=False,
                     output=z_rotated_ref, order=1, mode='constant', cval=0.0,
                     prefilter=False)
        scimg.rotate(input=z_rotated_ref, angle=psi, axes=(1, 0), reshape=False,
                     output=xz_rotated_ref, order=1, mode='constant', cval=0.0,
                     prefilter=False)
        scimg.rotate(input=xz_rotated_ref, angle=theta, axes=(2, 1),
                     reshape=False, output=zxz_rotated_ref, order=1,
                     mode='constant', cval=0.0, prefilter=False)

    return zxz_rotated_ref


def paste_reference(dataset: np.array, ref: np.array, center: tuple,
                    axis_in_tom_format=False):
    """
    Function that pastes a reference image ref into a dataset, by the rule
    that the maximum value between the two is preferred.

    :param dataset: dataset where the reference will be pasted.
    :param ref: the reference particle that wil be pasted in dataset
    :param center: coordinate in x, y, z format where the center of ref will be
     placed
    :param axis_in_tom_format: bool, if true, it is assumed that both ref and
    dataset are read in the coordinate system x, y, z (according to tom
    functions).
    :return: empty
    """
    cx, cy, cz = center
    ref_center = [int(sh * 0.5) for sh in ref.shape]
    index_0, index_1, index_2 = np.where(ref > 0)
    if axis_in_tom_format:
        r_cx, r_cy, r_cz = ref_center
        for x, y, z in zip(index_0, index_1, index_2):
            point_index = np.array(
                [x - r_cx + cx, y - r_cy + cy, z - r_cz + cz])
            if (np.array([0, 0, 0]) <= point_index).all() and (
                        point_index < np.array(dataset.shape)).all():
                dataset[x - r_cx + cx, y - r_cy + cy, z - r_cz + cz] = np.max(
                    [ref[x, y, z],
                     dataset[x - r_cx + cx, y - r_cy + cy, z - r_cz + cz]])
    else:
        r_cz, r_cy, r_cx = ref_center
        for z, y, x in zip(index_0, index_1, index_2):
            point_index = np.array(
                [z - r_cz + cz, y - r_cy + cy, x - r_cx + cx])
            if (np.array([0, 0, 0]) <= point_index).all() and (
                        point_index < np.array(dataset.shape)).all():
                dataset[z - r_cz + cz, y - r_cy + cy, x - r_cx + cx] = np.max(
                    [ref[z, y, x],
                     dataset[z - r_cz + cz, y - r_cy + cy, x - r_cx + cx]])
    return
