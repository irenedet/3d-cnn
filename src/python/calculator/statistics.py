import numpy as np


def precision_recall_calculator(motl_coords: np.array,
                                motl_clean_coords: np.array, radius: float):
    nclean = motl_clean_coords.shape[0]
    detected_clean = set()
    prec = []
    recall = []
    n = 0
    for motl_point in motl_coords:
        n = n + 1
        flag = 'undetected'
        for clean_point in motl_clean_coords:
            if flag == 'undetected':
                dist_vect = motl_point - clean_point
                dist = np.sqrt(np.sum(dist_vect ** 2))
                if ((dist <= radius) and (
                            tuple(clean_point) not in detected_clean)):
                    detected_clean = detected_clean | {tuple(clean_point)}
                    flag = 'detected'
        tp = len(detected_clean)
        prec += [tp / n]
        recall += [tp / nclean]
    return prec, recall, detected_clean


def precision_recall_calculator_and_detected(predicted_coordinates: np.array,
                                             predicted_values: list,
                                             true_coordinates: np.array,
                                             radius: float):
    total_true_points = true_coordinates.shape[0]
    detected_true = set()
    detected_predicted = set()
    value_detected_predicted = []
    undetected_predicted = set()
    value_undetected_predicted = []
    redundantly_detected_true = set()
    redundantly_detected_predicted = set()
    value_redudndantly_detected_predicted = []
    precision = []
    recall = []
    total_current_predicted_points = 0
    for score_value, predicted_point in zip(predicted_values,
                                            predicted_coordinates):
        total_current_predicted_points += 1
        flag = 'undetected'
        for true_point in true_coordinates:
            if flag == 'undetected':
                dist = np.linalg.norm(predicted_point - true_point)
                if ((dist <= radius) and (
                            tuple(true_point) not in detected_true)):
                    detected_true |= {tuple(true_point)}
                    detected_predicted |= {tuple(predicted_point)}
                    value_detected_predicted += [score_value]
                    flag = 'detected'
                elif ((dist <= radius) and (
                            tuple(true_point) in detected_true)):
                    redundantly_detected_true |= {tuple(true_point)}
                    redundantly_detected_predicted |= {tuple(predicted_point)}
                    value_redudndantly_detected_predicted += [score_value]
                    flag = 'redundantly_detected'
        if flag == "undetected":
            undetected_predicted |= {tuple(predicted_point)}
            value_undetected_predicted += [score_value]
        true_positives = len(detected_true)
        precision += [true_positives / total_current_predicted_points]
        recall += [true_positives / total_true_points]
    print("len(redundantly_detected_predicted) = ",
          len(redundantly_detected_predicted))
    return precision, recall, detected_true, detected_predicted, \
           undetected_predicted, value_detected_predicted, \
           value_undetected_predicted


def F1_score_calculator(prec: list, recall: list):
    F1_score = []

    for n in range(len(recall)):
        if prec[n] + recall[n] != 0:
            F1_score += [2.0 * prec[n] * recall[n] / float(prec[n] + recall[n])]
        else:
            F1_score += [0]

    return F1_score


def quadrature_calculator(x_points: list, y_points: list) -> float:
    """
    This function computes an approximate value of the integral of a real function f in an interval,
    using the trapezoidal rule.

    Input:
    x_points: is a list of points in the x axis (not necessarly ordered)
    y_points: is a list of points, such that y_points[n] = f(x_points[n]) for each n.
    """
    # sorted_y = [p for _, p in sorted(zip(x_points, y_points))]
    sorted_y = [p for _, p in
                sorted(list(zip(x_points, y_points)), key=lambda x: x[0])]
    n = len(y_points)
    sorted_x = sorted(x_points)

    trapezoidal_rule = [
        0.5 * (sorted_x[n + 1] - sorted_x[n]) * (sorted_y[n + 1] + sorted_y[n])
        for n in range(n - 1)]

    return float(np.sum(trapezoidal_rule))


def pr_auc_score(precision: list, recall: list) -> float:
    """
    This function computes an approximate value to the area
    under the precision-recall (PR) curve.
    """
    return quadrature_calculator(recall, precision)
