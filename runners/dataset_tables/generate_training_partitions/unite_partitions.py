import os

import h5py
from tqdm import tqdm

from constants.h5_internal_paths import RAW_SUBTOMOGRAMS, LABELED_SUBTOMOGRAMS


def add_tag(subtomo_name, tag):
    return tag + "_" + subtomo_name


def unite_partitions(output_partition, partition, tag_partition,
                     semantic_classes):
    with h5py.File(output_partition, 'a') as f:
        with h5py.File(partition, 'r') as p:
            subtomo_names_p = list(p[RAW_SUBTOMOGRAMS])
            n_subtomos = range(len(subtomo_names_p))

            print("Copying raw subtomos")
            for index, subtomo_name in zip(tqdm(n_subtomos), subtomo_names_p):
                tagged_subtomo_name = add_tag(subtomo_name, tag_partition)
                raw_internal_path_p = os.path.join(RAW_SUBTOMOGRAMS,
                                                   subtomo_name)

                raw_internal_path = os.path.join(RAW_SUBTOMOGRAMS,
                                                 tagged_subtomo_name)

                f[raw_internal_path] = p[raw_internal_path_p][:]

            print("Copying labeled subtomos")

            n_subtomos = range(len(subtomo_names_p))
            for index, subtomo_name in zip(tqdm(n_subtomos), subtomo_names_p):
                tagged_subtomo_name = add_tag(subtomo_name, tag_partition)
                for label in semantic_classes:
                    label_path = os.path.join(LABELED_SUBTOMOGRAMS, label)
                    label_internal_path_p = os.path.join(label_path,
                                                         subtomo_name)
                    label_internal_path = os.path.join(label_path,
                                                       tagged_subtomo_name)
                    f[label_internal_path] = p[label_internal_path_p][:]


if __name__ == '__main__':
    semantic_classes = ["fas"]
    tomo_names = [
        # "180426/027",
        # "180426/028",
        "180426/026",
        "180426/029",
        "180426/030",
        "180426/034",
        "180426/037",
        "180426/041",
        "180426/043",
        "180426/045"
    ]
    base_dir_DA = "/struct/mahamid/Irene/yeast/healthy"
    base_dir_2xf = "/struct/mahamid/Irene/scratch/3d-cnn/cross-validation/fas/original-training-data"
    output_partition = "strongly_labeled_min0.01_max1/random_partition/combined_DA_2xf.h5"

    partition_DA = "strongly_labeled_min0.01_max1/single_filter_64pix/G5_E0_R180_DArounds4/full_partition.h5"
    partition_2xf = "strongly_labeled_min0.01_max1/random_partition/2xfilter_64pix/random_partition.h5"

    tag_partition_DA = "DA"
    tag_partition_2xf = "2xf"

    partition_reldirs = [partition_2xf, partition_DA]
    tags = [tag_partition_2xf, tag_partition_DA]
    for tomo_name in tomo_names:
        print("Tomo", tomo_name)
        tomo_dir_DA = os.path.join(base_dir_DA, tomo_name)
        tomo_dir_2xf = os.path.join(base_dir_2xf, tomo_name)
        base_tomo_dirs = [tomo_dir_2xf, tomo_dir_DA]

        output_tomo_path = os.path.join(tomo_dir_DA, output_partition)
        print("output:", output_tomo_path)
        output_tomo_dir = os.path.dirname(output_tomo_path)
        os.makedirs(output_tomo_dir, exist_ok=True)

        partitions = [os.path.join(tomo_dir, partition) for tomo_dir, partition
                      in zip(base_tomo_dirs, partition_reldirs)]

        for tag, partition in zip(tags, partitions):
            print(tag, partition)
            unite_partitions(output_tomo_path, partition, tag, semantic_classes)
