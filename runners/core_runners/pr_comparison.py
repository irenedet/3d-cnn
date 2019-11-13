import matplotlib.pyplot as plt
import numpy as np

from os import makedirs
from os.path import join

from src.python.calculator.statistics import pr_auc_score, \
    F1_score_calculator, precision_recall_calculator_and_detected
from src.python.peak_toolbox.utils import read_motl_coordinates_and_values
from src.python.filewriters.csv import motl_writer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="Path for output folders and motls.")
    parser.add_argument("--motl_ref",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to the motl file for reference.")
    parser.add_argument("--motl_comp",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to the motl file for comparison.")
    parser.add_argument("--radius",
                        default=None,
                        type=int or float,
                        required=True,
                        help="Precision radius for comparison of two motls.")

    output_dir = parser.output_dir
    path_to_motl_predicted = parser.motl_ref
    path_to_motl_true = parser.motl_comp
    radius = parser.radius

    figures_dir = join(output_dir, "figures")
    makedirs(name=figures_dir, exist_ok=True)

    dataset_table = "None"
    threshold = -np.inf
    tight_threshold = 0
    class_name = 'ribo'
    z_shift = 0
    y_shift = 0
    x_shift = 0
    x_dim, y_dim, z_dim = 928, 928, 500

    predicted_values, predicted_coordinates = read_motl_coordinates_and_values(
        path_to_motl=path_to_motl_predicted)

    true_values, true_coordinates = read_motl_coordinates_and_values(
        path_to_motl=path_to_motl_true)
    unique_peaks_number = len(predicted_values)

    print("all particles will be considered for testing...")
    true_coordinates_test = true_coordinates
    predicted_coordinates_test = predicted_coordinates
    predicted_values_test = predicted_values
    shift_vector = np.array([x_shift, y_shift, z_shift])
    predicted_coordinates_test = [np.array(p) + shift_vector for p in
                                  predicted_coordinates_test]
    predicted_coordinates_test = np.array(predicted_coordinates_test)

    precision, recall, detected_true, detected_predicted, \
    undetected_predicted, value_detected_predicted, \
    value_undetected_predicted, redundantly_detected_predicted, \
    value_redudndantly_detected_predicted = \
        precision_recall_calculator_and_detected(
            predicted_coordinates_test,
            predicted_values_test,
            true_coordinates_test,
            radius=radius)

    F1_score = F1_score_calculator(precision, recall)
    if len(F1_score) > 0:
        max_F1 = np.max(F1_score)
        optimal_peak_number = np.min(np.where(F1_score == max_F1)[0])
    else:
        max_F1 = np.nan
        optimal_peak_number = np.nan

    auPRC = pr_auc_score(precision=precision, recall=recall)
    print("auPRC = ", auPRC, "and max_F1 = ", max_F1)

    path_to_detected_predicted = join(output_dir, "detected")
    path_to_undetected_predicted = join(output_dir, "undetected")
    makedirs(name=path_to_detected_predicted, exist_ok=True)
    makedirs(name=path_to_undetected_predicted, exist_ok=True)

    motl_writer(path_to_output_folder=path_to_detected_predicted,
                list_of_peak_coords=detected_predicted,
                list_of_peak_scores=value_detected_predicted,
                in_tom_format=True)
    motl_writer(path_to_output_folder=path_to_undetected_predicted,
                list_of_peak_coords=undetected_predicted,
                list_of_peak_scores=value_undetected_predicted,
                in_tom_format=True)

    print("peak threshold set to ", threshold)

    thresholded_predicted_indices = \
        np.where(np.array(predicted_values_test) >= threshold)[0]
    thresholded_predicted_values = [predicted_values_test[index] for
                                    index in thresholded_predicted_indices]

    thresholded_detected_predicted_indices = \
        np.where(np.array(value_detected_predicted) >= threshold)[0]
    value_detected_predicted_range = [value_detected_predicted[index] for index
                                      in
                                      thresholded_detected_predicted_indices]

    thresholded_undetected_predicted_indices = \
        np.where(np.array(value_undetected_predicted) >= threshold)[0]
    value_undetected_predicted_range = \
        [value_undetected_predicted[index] for index in
         thresholded_undetected_predicted_indices]

    tight_thresholded_detected_predicted_indices = \
        np.where(np.array(value_detected_predicted) >= tight_threshold)[0]

    value_detected_predicted_tight = \
        [value_detected_predicted[index] for index in
         tight_thresholded_detected_predicted_indices]

    tight_thresholded_undetected_predicted_indices = \
        np.where(np.array(value_undetected_predicted) >= tight_threshold)[0]

    value_undetected_predicted_tight = [
        value_undetected_predicted[index] for index in
        tight_thresholded_undetected_predicted_indices]

    # First plot
    plt.figure(1)
    plt.hist(thresholded_predicted_values, bins=45,
             label="thresholded predicted")
    plt.xlabel("score value")
    plt.ylabel("frequency")
    plt.title(str(len(thresholded_predicted_indices)) + " peaks")
    plt.legend()
    plt.gcf()
    fig_name = join(figures_dir, "histogram_test_set.png")
    plt.savefig(fname=fig_name,
                format="png")
    plt.close()

    plt.figure(3)
    plt.hist(value_detected_predicted_range, bins=45, label="true positives",
             fc=(0, 0, 1, 0.5))
    plt.hist(value_undetected_predicted_range, bins=45, label="false positives",
             fc=(1, 0, 0, 0.5))
    plt.xlabel("score value")
    plt.ylabel("frequency")
    plt.title(str(len(thresholded_undetected_predicted_indices)) + " peaks")
    plt.legend()
    plt.gcf()
    fig_name = join(figures_dir, "histogram-detected-undetected_testset.png")
    plt.savefig(fname=fig_name, format="png")
    plt.close()

    plt.figure(4)
    plt.hist(value_detected_predicted_tight, bins=45, label="true positives",
             fc=(0, 0, 1, 0.5))
    plt.hist(value_undetected_predicted_tight, bins=45, label="false positives",
             fc=(1, 0, 0, 0.5))
    plt.xlabel("score value")
    plt.ylabel("frequency")
    plt.title(str(len(tight_thresholded_detected_predicted_indices) +
                  len(tight_thresholded_undetected_predicted_indices))
              + " peaks, threshold on score = " + str(tight_threshold))
    plt.legend()
    plt.gcf()
    fig_name = join(figures_dir,
                    "histogram-detected-undetected_thresholded_testset.png")
    plt.savefig(fname=fig_name, format="png")
    plt.close()

    plt.figure(6)
    pr_legend_str = "auPRC = " + str(round(auPRC, 4))
    f1_legend_str = "(max_F1, best_peaks) = (" + str(
        round(max_F1, 4)) + ", " + str(
        optimal_peak_number) + ")"
    title_str = str(len(predicted_coordinates_test)) + " peaks"
    plt.plot(F1_score, label=f1_legend_str)
    plt.xlabel("number of peaks")
    plt.ylabel("F1 score")
    plt.title(title_str)
    plt.legend()
    plt.gcf()
    fig_name = join(figures_dir, "f1_score_" + title_str + ".png")
    plt.savefig(fname=fig_name, format="png")
    plt.close()

    plt.figure(7)
    plt.plot(recall, precision, label=pr_legend_str)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title(title_str)
    plt.legend()
    plt.gcf()
    fig_name = join(figures_dir, "pr_" + title_str + ".png")
    plt.savefig(fname=fig_name, format="png")
    plt.close()
    print("All plots saved in ", figures_dir)

    statistics_label = label_name + "_pr_radius_" + str(radius)
    print("statistics_label =", statistics_label)
