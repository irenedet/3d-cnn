import h5py
from os.path import join
from shutil import copyfile
from os import makedirs

from random import shuffle

tomos = [
    # '180426/004',
    # '180426/005',
    # '180426/021',
    # '181119/002',
    '181119/030',
    # '181126/002',
    # '181126/012',
]
# /scratch/trueba/3d-cnn/cross-validation/training-data/181119/002/multi_class/strongly_labeled_0.002/full_partition.h5
# /scratch/trueba/3d-cnn/cross-validation/training-data/181119/030/multi_class/strongly_labeled_0.002/full_partition.h5
# /scratch/trueba/3d-cnn/cross-validation/training-data/181126/002/multi_class/strongly_labeled_0.002/full_partition.h5
# /scratch/trueba/3d-cnn/cross-validation/training-data/181126/012/multi_class/strongly_labeled_0.002/full_partition.h5

fractions_number = 5
semantic_classes = ['ribo', 'fas', 'memb']
base_dir = "/scratch/trueba/3d-cnn/cross-validation/training-data/"
tail_dir = "multi_class/strongly_labeled_0.002"

copy_to_global = "/scratch/trueba/3d-cnn/cross-validation/training-fractions"
for tomo in tomos:
    input_dir = join(base_dir, tomo)
    input_dir = join(input_dir, tail_dir)
    input_file = join(input_dir, "full_partition.h5")
    output_folder = input_dir
    makedirs(output_folder, exist_ok=True)
    print("Starting to fraction ", input_dir)
    with h5py.File(input_file, 'r') as f:
        subtomo_names = list(f['volumes/raw'])
        print("Number of subtomos", len(subtomo_names))
        n = len(subtomo_names) // fractions_number
        print("Subtomos per fraction =", n)
        shuffle(subtomo_names)
        for fraction in range(fractions_number):
            fraction_name = "fraction_" + str(fraction) + ".h5"
            output_file = join(output_folder, fraction_name)
            with h5py.File(output_file, "w") as f_frac:
                for subtomo_name in subtomo_names[
                                    fraction * n:(fraction + 1) * n]:
                    print("Subtomo in fraction", fraction, ":", subtomo_name)
                    volume_internal_path = join('volumes/raw', subtomo_name)
                    f_frac[volume_internal_path] = f[volume_internal_path][:]
                    for semantic_class in semantic_classes:
                        semantic_class_internal_path = join('volumes/labels',
                                                            semantic_class)
                        label_internal_path = join(semantic_class_internal_path,
                                                   subtomo_name)
                        f_frac[label_internal_path] = f[label_internal_path][:]

            print("Now copying files to cv locations:")
            src = output_file
            print("src", src)
            for training_set in range(fractions_number):
                training_fraction_name = "training_fraction_" + str(
                    training_set)
                tomo_fraction_iteration = tomo[:6] + "_" + tomo[-3:] + \
                                          "_fraction_" + str(fraction) + ".h5"
                dst_tomo = join(copy_to_global, training_fraction_name)
                makedirs(dst_tomo, exist_ok=True)
                dst = join(dst_tomo, tomo_fraction_iteration)
                print(dst)
                copyfile(src, dst)
    print("Done with", tomo)
