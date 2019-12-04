import h5py
from os.path import join
from shutil import copyfile
from os import makedirs

from random import shuffle

tomos = ["181126/025"]
# tomos = ["180426/024", "181126/025"]
# sample_types = ['healthy', 'ED']
org_files = ["/struct/mahamid/Irene/yeast/ED/181126/025/eman_filt_eman_filt_tomo_partition.h5"]
fractions_number = 5
src_global_dir = "/struct/mahamid/Irene/yeast"
dst_global_dir = "/scratch/trueba/3d-cnn/cross-validation/full_test_datasets"
for tomo, input_file in zip(tomos, org_files):
    with h5py.File(input_file, 'r') as f:
        subtomo_names = list(f['volumes/raw'])
        print("Number of subtomos", len(subtomo_names))
        for fraction in range(fractions_number):
            print("Starting to copy data to fraction ", fraction)
            output_folder = join(dst_global_dir, tomo)
            fraction_name = "fraction_" + str(fraction)
            output_folder = join(output_folder, fraction_name)
            makedirs(output_folder, exist_ok=True)

            output_file = join(output_folder, "full_partition.h5")
            with h5py.File(output_file, "w") as f_frac:
                for subtomo_name in subtomo_names:
                    print("Subtomo in fraction", fraction, ":", subtomo_name)
                    volume_internal_path = join('volumes/raw', subtomo_name)
                    f_frac[volume_internal_path] = f[volume_internal_path][:]
    print("Done with", tomo)

