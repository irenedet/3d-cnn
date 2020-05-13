import argparse
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from constants.dataset_tables import ModelsTableHeader, DatasetTableHeader
from file_actions.writers.csv import motl_writer
from file_actions.writers.csv import write_statistics
from networks.utils import build_prediction_output_dir
from performance.statistics_utils import pr_auc_score, \
    f1_score_calculator, precision_recall_calculator
from tomogram_utils.peak_toolbox.utils import read_motl_coordinates_and_values

parser = argparse.ArgumentParser()
parser.add_argument("-yaml_file", "--yaml_file", help="yaml_file", type=str)
parser.add_argument("-tomos_set", "--tomos_set",
                    help="tomos to apply the analysis in dataset table",
                    type=int)
args = parser.parse_args()
tomos_set = args.tomos_set
yaml_file = args.yaml_file
config = yaml.safe_load(open(yaml_file))

tomo_list = config['tomos_sets'][tomos_set]['test_list']
unet_hyperparameters = config['unet_hyperparameters']

if 'model_name' in config['tomos_sets'][tomos_set].keys():
    model_name = config['tomos_sets'][tomos_set]['model_name']
else:
    model_name = unet_hyperparameters['model_name']

label_name = unet_hyperparameters['label_name']
segmentation_label = label_name + "_" + model_name
overlap = config['overlap']
box_shape = config['box_shape']
if isinstance(box_shape, int):
    box_shape = [box_shape, box_shape, box_shape]
else:
    box_shape = tuple([int(elem) for elem in reversed(box_shape)])

assert len(box_shape) == 3, "Invalid box_shape"

class_number = config['reconstruction']['class_number']

ModelsHeader = ModelsTableHeader()
models_table = unet_hyperparameters['models_table']
models_df = pd.read_csv(models_table,
                        dtype={ModelsHeader.model_name: str,
                               ModelsHeader.semantic_classes: str})
model_df = models_df[models_df[ModelsHeader.model_name] == model_name]
semantic_classes = model_df.iloc[0]['segmentation_names'].split(',')
semantic_class = semantic_classes[class_number]
motl_parameters = config['motl_parameters']
filtering_mask = config['motl_parameters']['filtering_mask']

for tomo_name in tomo_list:
    output_dir = config['reconstruction']['reconstruction_path']
    output_dir = build_prediction_output_dir(base_output_dir=output_dir,
                                             label_name=label_name,
                                             model_name=model_name,
                                             tomo_name=tomo_name,
                                             semantic_class=semantic_class)

    output_dir = os.path.join(output_dir, "in_" + filtering_mask)
    motls_in_dir = [file for file in os.listdir(output_dir) if 'motl_' in file]
    assert len(motls_in_dir) == 1, "only one motive list can be filtered."
    path_to_motl_predicted = os.path.join(output_dir, motls_in_dir[0])

    dataset_table = config['dataset_table']
    clean_motl = config['performance_parameters']['clean_motl']
    statistics_file = config['performance_parameters']['statistics_file']

    DTHeader = DatasetTableHeader(clean_motl=clean_motl)
    df = pd.read_csv(dataset_table)
    df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)
    tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
    radius = config['performance_parameters']['pr_tolerance_radius']
    box_shape = config['box_shape']
    if isinstance(box_shape, int):
        box_shape = [box_shape, box_shape, box_shape]
    else:
        box_shape = tuple([int(elem) for elem in reversed(box_shape)])

    assert len(box_shape) == 3, "Invalid box_shape"

    z_shift = int(tomo_df.iloc[0][DTHeader.z_shift])
    y_shift = int(tomo_df.iloc[0][DTHeader.y_shift])
    x_shift = int(tomo_df.iloc[0][DTHeader.x_shift])
    x_dim = int(tomo_df.iloc[0][DTHeader.x_dim])
    y_dim = int(tomo_df.iloc[0][DTHeader.y_dim])
    z_dim = int(tomo_df.iloc[0][DTHeader.z_dim])

    path_to_motl_true = tomo_df.iloc[0][DTHeader.clean_motls]

    print("")
    print("path_to_motl_true = ", path_to_motl_true)
    print("output_dir = ", output_dir)
    dataset_shape = (z_dim, y_dim, x_dim)
    subtomo_shape = tuple([sh - overlap for sh in box_shape])

    class_number = config['reconstruction']['class_number']
    class_name = semantic_classes[class_number]

    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    predicted_values, predicted_coordinates = read_motl_coordinates_and_values(
        path_to_motl=path_to_motl_predicted)

    true_values, true_coordinates = read_motl_coordinates_and_values(
        path_to_motl=path_to_motl_true)
    unique_peaks_number = len(predicted_values)

    shift_vector = np.array([x_shift, y_shift, 0])
    predicted_coordinates = [np.array(p) + shift_vector for p in
                             predicted_coordinates]
    predicted_coordinates = np.array(predicted_coordinates)

    prec, recall, tp_true, tp_pred, fp_pred, tp_pred_scores, fp_pred_scores, *_ = \
        precision_recall_calculator(
            predicted_coordinates=predicted_coordinates,
            value_predicted=predicted_values,
            true_coordinates=true_coordinates,
            radius=radius)

    F1_score = f1_score_calculator(prec, recall)

    if len(F1_score) > 0:
        max_F1 = np.max(F1_score)
        optimal_peak_number = np.min(np.where(F1_score == max_F1)[0])
    else:
        max_F1 = 0
        optimal_peak_number = np.nan

    auPRC = pr_auc_score(precision=prec, recall=recall)
    print("auPRC = ", auPRC, "and max_F1 = ", max_F1)

    output_dir = os.path.join(output_dir, "pr_radius_" + str(radius))
    path_to_detected_predicted = join(output_dir, "detected")
    path_to_undetected_predicted = join(output_dir, "undetected")

    os.makedirs(path_to_detected_predicted, exist_ok=True)
    os.makedirs(path_to_undetected_predicted, exist_ok=True)

    motl_writer(path_to_output_folder=path_to_detected_predicted,
                list_of_peak_coords=tp_pred,
                list_of_peak_scores=tp_pred_scores,
                in_tom_format=True)
    motl_writer(path_to_output_folder=path_to_undetected_predicted,
                list_of_peak_coords=fp_pred,
                list_of_peak_scores=fp_pred_scores,
                in_tom_format=True)

    plt.figure(1)
    plt.hist(predicted_values, bins=45, label="predicted")
    plt.xlabel("score value")
    plt.ylabel("frequency")
    plt.title(str(len(predicted_values)) + " peaks")
    plt.legend()
    plt.gcf()
    fig_name = join(figures_dir, "histogram_predicted.png")
    plt.savefig(fname=fig_name, format="png")

    plt.figure(2)
    plt.hist(tp_pred_scores, bins=45, label="true positives",
             fc=(0, 0, 1, 0.5))
    plt.hist(fp_pred_scores, bins=45, label="false positives",
             fc=(1, 0, 0, 0.5))
    plt.xlabel("score value")
    plt.ylabel("frequency")
    plt.title(str(len(fp_pred)) + " peaks")
    plt.legend()
    plt.gcf()
    fig_name = join(figures_dir, "histogram-detected-undetected.png")
    plt.savefig(fname=fig_name, format="png")

    plt.figure(3)
    pr_legend_str = "auPRC = " + str(round(auPRC, 4))
    f1_legend_str = "(max_F1, best_peaks) = (" + str(
        round(max_F1, 4)) + ", " + str(optimal_peak_number) + ")"
    title_str = str(len(predicted_coordinates)) + " peaks"
    plt.plot(F1_score, label=f1_legend_str)
    plt.xlabel("number of peaks")
    plt.ylabel("F1 score")
    plt.title(title_str)
    plt.legend()
    plt.gcf()
    fig_name = join(figures_dir, "f1_score_" + title_str + ".png")
    plt.savefig(fname=fig_name, format="png")

    plt.figure(4)
    plt.plot(recall, prec, label=pr_legend_str)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title(title_str)
    plt.legend()
    plt.gcf()
    fig_name = join(figures_dir, "pr_" + title_str + ".png")
    plt.savefig(fname=fig_name, format="png")

    statistics_label = segmentation_label + "_pr_radius_" + str(radius)

    write_statistics(statistics_file=statistics_file,
                     statistics_label= "auPRC_" + statistics_label,
                     tomo_name=tomo_name,
                     stat_measure=round(auPRC, 4))

    write_statistics(statistics_file=statistics_file,
                     statistics_label= "F1_" + statistics_label,
                     tomo_name=tomo_name,
                     stat_measure=round(max_F1, 4))
