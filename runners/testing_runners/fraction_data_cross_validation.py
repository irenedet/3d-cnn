import h5py
from os.path import join
from random import shuffle

input_h5 = "/struct/mahamid/Irene/yeast/healthy/180426/021/G_sigma1_non_sph/train_and_test_partitions/full_partition.h5"
output_folder = "/struct/mahamid/Irene/yeast/healthy/180426/021/G_sigma1_non_sph/train_and_test_partitions/"
fractions_number = 5
semantic_classes = ['fas', 'memb', 'ribo']

with h5py.File(input_h5, 'r') as f:
    subtomo_names = list(f['volumes/raw'])
    n = len(subtomo_names) // fractions_number
    shuffle(subtomo_names)
    for fraction in range(fractions_number):
        fraction_name = "fraction_" + str(fraction) + ".h5"
        output_file = join(output_folder, fraction_name)
        with h5py.File(output_file, "w") as f_frac:
            for subtomo_name in subtomo_names[fraction * n:(fraction + 1) * n]:
                volume_internal_path = join('volumes/raw', subtomo_name)
                f_frac[volume_internal_path] = f[volume_internal_path][:]
                for semantic_class in semantic_classes:
                    semantic_class_internal_path = join('volumes/labels',
                                                        semantic_class)
                    label_internal_path = join(semantic_class_internal_path,
                                               subtomo_name)
                    f_frac[label_internal_path] = f[label_internal_path][:]
