import numpy as np
import h5py
from os.path import join
import os.path
from os import makedirs
import pandas as pd

from src.python.filereaders.datasets import load_dataset
from src.python.peak_toolbox.utils import read_motl_coordinates_and_values
from src.python.naming import h5_internal_paths
from src.python.tensors.actions import crop_window_around_point


def generate_classification_trainingset(path_to_output_h5: str,
                                        path_to_dataset: str,
                                        motl_path: str, label: str,
                                        box_side: int or tuple):
    if type(box_side) is int:
        crop_shape = (box_side, box_side, box_side)
    else:
        crop_shape = box_side
    values, coordinates = read_motl_coordinates_and_values(
        path_to_motl=motl_path)
    dataset = load_dataset(path_to_dataset=path_to_dataset)

    if os.path.isfile(path_to_output_h5):
        mode = 'a'
    else:
        mode = 'w'

    makedirs(name=os.path.dirname(path_to_output_h5), exist_ok=True)
    with h5py.File(path_to_output_h5, mode) as f:
        internal_path = h5_internal_paths.LABELED_SUBTOMOGRAMS
        internal_path = join(internal_path, label)
        for point in coordinates:
            x, y, z = [int(entry) for entry in point]
            subtomo_name = "subtomo_" + str(point)
            subtomo = crop_window_around_point(input_array=dataset,
                                               crop_shape=crop_shape,
                                               window_center=(z, y, x))
            subtomo_path = join(internal_path, subtomo_name)
            f[subtomo_path] = subtomo[:]
    return path_to_output_h5


def generate_classification_trainingset_per_tomo(dataset_table: str,
                                                 tomo_name: str,
                                                 semantic_classes: list,
                                                 path_to_output_h5: str,
                                                 box_side: int or tuple):
    df = pd.read_csv(dataset_table)
    df['tomo_name'] = df['tomo_name'].astype(str)
    tomo_df = df[df['tomo_name'] == tomo_name]

    for semantic_class in semantic_classes:
        motl_label = "path_to_motl_clean_" + semantic_class
        motl_path = tomo_df.iloc[0][motl_label]
        path_to_dataset = tomo_df.iloc[0]['eman2_filetered_tomo']
        generate_classification_trainingset(path_to_output_h5, path_to_dataset,
                                            motl_path, semantic_class, box_side)
    return


def get_subtomos_and_labels(path_to_output_h5, label):
    with h5py.File(path_to_output_h5, 'r') as f:
        internal_path = h5_internal_paths.LABELED_SUBTOMOGRAMS
        internal_path = join(internal_path, label)
        subtomos_names = list(f[internal_path])
        subtomos_number = len(subtomos_names)
        subtomos_list = []
        for subtomo_name in subtomos_names:
            subtomo_path = join(internal_path, subtomo_name)
            subtomo = f[subtomo_path][:]
            subtomos_list.append(subtomo)
        labels = np.ones(subtomos_number)
    return subtomos_list, labels


def fill_multiclass_labels(semantic_classes, co_labeling_dict, labels_data):
    # format of co_labeling_dict: if class1 \subset class2 then class2 in co_labeling[class1]
    for class_index, semantic_class in enumerate(semantic_classes):
        for other_index, other_class in enumerate(semantic_classes):
            if other_class in co_labeling_dict[semantic_class]:
                print(semantic_class, " is a subset of ", other_class)
                where_semantic_class = np.where(labels_data[class_index, :] > 0)
                labels_data[other_index, where_semantic_class] = np.ones(
                    (1, len(where_semantic_class)))
    return labels_data


def generate_multiclass_labels(semantic_classes, path_to_output_h5):
    subtomos_data = []
    classes_number = len(semantic_classes)
    labels_data = np.zeros((classes_number, 0))
    for class_number, semantic_class in enumerate(semantic_classes):
        subtomos_list, labels = get_subtomos_and_labels(path_to_output_h5,
                                                        semantic_class)
        labels_semantic_class = np.zeros((classes_number, len(labels)))
        labels_semantic_class[class_number, :] = labels

        subtomos_data += subtomos_list
        labels_data = np.concatenate((labels_data, labels_semantic_class),
                                     axis=1)
    return subtomos_data, labels_data


box_side = 64
path_to_output_h5 = "/scratch/trueba/3Dclassifier/liang_data/training_data/200/training_set.h5"
semantic_classes = ['70S', '50S']
multi_label = True
dataset_table = "/struct/mahamid/Irene/liang_data/multiclass/liang_data_multiclass.csv"
tomo_name = '200'

generate_classification_trainingset_per_tomo(dataset_table,
                                             tomo_name,
                                             semantic_classes,
                                             path_to_output_h5,
                                             box_side)

# if class1 \subset class2 then class2 in co_labeling[class1]
co_labeling_dict = {'70S': ['50S'], '50S': []}

subtomos_data, labels_data = generate_multiclass_labels(semantic_classes,
                                                        path_to_output_h5)

if multi_label:
    # Check for dependencies (co-labeling):
    print("Multiple labels are allowed for a particle: multi-label case.")
    labels_data = fill_multiclass_labels(semantic_classes, co_labeling_dict,
                                         labels_data)
else:
    print("Only one class per particle: multi-class case.")
