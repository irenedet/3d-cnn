import os

import h5py
import pandas as pd
from tqdm import tqdm

from constants.dataset_tables import DatasetTableHeader
from constants.h5_internal_paths import RAW_SUBTOMOGRAMS, LABELED_SUBTOMOGRAMS
from file_actions.readers.tomograms import load_tomogram
from tensors.actions import crop_window_around_point
from tomogram_utils.coordinates_toolbox.subtomos import get_coord_from_name


def add_labels2partition(partition, semantic_name, label_mask,
                         crop_shape) -> None:
    with h5py.File(partition, 'a') as f:
        subtomo_names = list(f[RAW_SUBTOMOGRAMS])
        n_subtomos = range(len(subtomo_names))

        print("Cropping label subtomos")
        for index, subtomo_name in zip(tqdm(n_subtomos), subtomo_names):
            window_center = get_coord_from_name(subtomo_name=subtomo_name)

            subtomo_label_data = crop_window_around_point(
                input_array=label_mask,
                crop_shape=crop_shape,
                window_center=window_center)
            label_internal_path = os.path.join(LABELED_SUBTOMOGRAMS,
                                               semantic_name)
            subtomo_label_path = os.path.join(label_internal_path,
                                              subtomo_name)
            f[subtomo_label_path] = subtomo_label_data
    return


global_base_dir = "/struct/mahamid/Irene/shrec2020/data/partitions/training"
partition_tail_name = "all_particles/strongly_labeled_min0.01_max1/random_partition.h5"
dataset_table = "/struct/mahamid/Irene/shrec2020/data/table.csv"
partition_name = 'train_partition'
tomo_names = [str(val) for val in range(2, 9)]
semantic_classes = [str(num) for num in range(1, 13)]
print("semantic_classes =", semantic_classes)
crop_shape = (64, 64, 64)

DTHeader = DatasetTableHeader(semantic_classes=semantic_classes,
                              partition_name=partition_name)

dataset_table_df = pd.read_csv(dataset_table)
dataset_table_df['tomo_name'] = dataset_table_df['tomo_name'].astype(str)
for tomo_name in tomo_names:
    tomo_df = dataset_table_df[dataset_table_df['tomo_name'] == tomo_name]
    base_dir = os.path.join(global_base_dir, tomo_name)
    partition = os.path.join(base_dir, partition_tail_name)
    for semantic_class, mask_column in zip(semantic_classes,
                                           DTHeader.masks_names):
        print(semantic_class, mask_column)
        label_path = tomo_df.iloc[0][mask_column]
        label_mask = load_tomogram(path_to_dataset=label_path)
        add_labels2partition(partition=partition, semantic_name=semantic_class,
                             label_mask=label_mask, crop_shape=crop_shape)
