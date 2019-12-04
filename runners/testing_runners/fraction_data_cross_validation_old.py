import h5py
from os.path import join
from random import shuffle

tomos = [
    "181119/002",
    "181119/030",
    "181126/002",
    "181126/012",
    "181126/025",
]
#     "190301/003",
#     #"190301/005",
#     "190301/009",
#     "190301/012",
#     "190301/016",
#     "190301/022",
# tomos = [
#     "190301/028",
#     "190301/032",
#     "190301/035",
#     "190301/037",
#     "190301/043",
#     #"190301/045",
# ]

fractions_number = 5
semantic_classes = ['fas']
base_dir = "/struct/mahamid/Irene/yeast/ED"
tail_dir = "fas_class/train_and_test_partitions/full_partition.h5"
for tomo in tomos:
    input_h5 = join(base_dir, tomo)
    input_h5 = join(input_h5, tail_dir)
    output_folder = join(base_dir, tomo)
    # output_folder = join(output_folder, "training_partition")
    output_folder = join(output_folder, "fas_class/")
    output_folder = join(output_folder, "train_and_test_partitions")

    with h5py.File(input_h5, 'r') as f:
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
                    volume_internal_path = join('volumes/raw', subtomo_name)
                    f_frac[volume_internal_path] = f[volume_internal_path][:]
                    for semantic_class in semantic_classes:
                        semantic_class_internal_path = join('volumes/labels',
                                                            semantic_class)
                        label_internal_path = join(semantic_class_internal_path,
                                                   subtomo_name)
                        f_frac[label_internal_path] = f[label_internal_path][:]
    print("Done with", tomo)
