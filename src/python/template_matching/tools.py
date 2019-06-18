from src.python.filereaders.datasets import load_dataset
import h5py
from src.python.naming import h5_internal_paths
from os.path import join
from src.python.datasets.transformations import rotate_ref
from src.python.calculator.math import radians2degrees


def create_template_catalogue(output_path: str, reference_file: str,
                              angles_file: str, in_degrees=False):
    reference = load_dataset(reference_file)
    angles = load_dataset(angles_file)
    if in_degrees:
        zxz_angles_in_degrees = angles
    else:
        zxz_angles_in_degrees = radians2degrees(angles)
    with h5py.File(output_path, 'w') as f:
        for index, angle in enumerate(list(zxz_angles_in_degrees)):
            rotation_name = str(index)
            rotated_reference = \
                rotate_ref(ref=reference,
                           zxz_angles_in_degrees=angle,
                           mode="nearest")
            internal_path = join(h5_internal_paths.RAW_SUBTOMOGRAMS,
                                 rotation_name)
            f[internal_path] = rotated_reference[:]
    return
