import h5py
import numpy as np
from os.path import join
from src.python.tensors.actions import crop_window_around_point


def pad_dataset(dataset, shape_to_crop_zyx):
    tomo_shape = dataset.shape
    padding = [(0, box_size - (tomo_size % box_size)) for tomo_size, box_size
               in zip(tomo_shape, shape_to_crop_zyx)]
    padded_dataset = np.pad(array=dataset, pad_width=padding, mode="reflect")
    return padded_dataset


def get_particle_coordinates_grid(dataset_shape, shape_to_crop_zyx):
    particle_coordinates = []
    nz_coords, ny_coords, nx_coords = [tomo_dim // box_size for
                                       tomo_dim, box_size in
                                       zip(dataset_shape, shape_to_crop_zyx)]
    for z in range(nz_coords):
        for y in range(ny_coords):
            for x in range(nx_coords):
                particle_coordinates += [
                    np.array(shape_to_crop_zyx) * np.array([z, y, x])
                    + np.array(shape_to_crop_zyx) // 2]
    return particle_coordinates


def write_subtomograms_from_dataset(output_path, padded_dataset,
                                    particles_coordinates, shape_to_crop_zyx):
    with h5py.File(output_path, 'w') as f:
        for particle_index, particle_center in enumerate(particles_coordinates):
            subtomo_name = "subtomo_" + "_" + str(particle_center)
            subtomo_h5_internal_path = join('volumes/subtomos', subtomo_name)
            subtomo_data = crop_window_around_point(input=padded_dataset,
                                                    shape_to_crop_zyx=shape_to_crop_zyx,
                                                    window_center_zyx=particle_center)
            f[subtomo_h5_internal_path] = subtomo_data


def split_tomo_into_subtomos(dataset, output_path: str,
                             shape_to_crop_zyx=(128, 128, 128)):
    padded_dataset = pad_dataset(dataset, shape_to_crop_zyx)
    print("The padded dataset has shape", padded_dataset.shape)
    particles_coordinates = get_particle_coordinates_grid(padded_dataset.shape,
                                                          shape_to_crop_zyx)
    write_subtomograms_from_dataset(output_path, padded_dataset,
                                    particles_coordinates, shape_to_crop_zyx)


path_to_raw = '/scratch/trueba/cnn/004/4bin/cnn/rawtomogram/001_bin4_subregion0-0-380-927-927-600.hdf'
with h5py.File(path_to_raw, 'r') as f:
    raw_dataset = f['MDF/images/0/image'][:]

folder_path = "/scratch/trueba/3d-cnn/evaluating_data/"
output_path = join(folder_path, "tomo004_in_subtomos_128side.h5")
split_tomo_into_subtomos(dataset=raw_dataset, output_path=output_path,
                         shape_to_crop_zyx=(128, 128, 128))


