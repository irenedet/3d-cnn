import numpy as np


def get_clean_points_close2point(point, clean, radius):
    close_to_point = list()
    for clean_p in clean:
        dist = np.linalg.norm(clean_p - point)
        if dist <= radius:
            close_to_point.append(clean_p)
    return close_to_point


def precision_recall_calculator_and_detected(
        predicted_coordinates: np.array or list,
        value_predicted: list,
        true_coordinates: np.array or list,
        radius: float):
    true_coordinates = list(true_coordinates)
    predicted_coordinates = list(predicted_coordinates)
    detected_true = list()
    predicted_true_positives = list()
    predicted_false_positives = list()
    predicted_redundant = list()
    value_predicted_true_positives = list()
    value_predicted_false_positives = list()
    value_predicted_redundant = list()
    precision = list()
    recall = list()
    total_true_points = len(true_coordinates)
    for value, point in zip(value_predicted, predicted_coordinates):
        close_to_point = get_clean_points_close2point(point, true_coordinates,
                                                      radius)
        if len(close_to_point) > 0:
            flag = "true_positive_candidate"
            flag_tmp = "not_redundant_yet"
            for clean_p in close_to_point:
                if flag == "true_positive_candidate":
                    if tuple(clean_p) not in detected_true:
                        detected_true.append(tuple(clean_p))
                        flag = "true_positive"
                    else:
                        flag_tmp = "redundant_candidate"
                else:
                    print(point, "is already flagged as true positive")
            if flag == "true_positive":
                predicted_true_positives.append(tuple(point))
                value_predicted_true_positives.append(value)
            elif flag == "true_positive_candidate" and \
                            flag_tmp == "redundant_candidate":
                predicted_redundant.append(tuple(point))
                value_predicted_redundant.append(value)
            else:
                print("This should never happen!")
        else:
            print("len(close_to_point) = ", len(close_to_point))
            predicted_false_positives.append(tuple(point))
            value_predicted_false_positives.append(value)
        true_positives_total = len(predicted_true_positives)
        false_positives_total = len(predicted_false_positives)
        total_current_predicted_points = true_positives_total + \
                                         false_positives_total
        precision += [true_positives_total / total_current_predicted_points]
        recall += [true_positives_total / total_true_points]
    return precision, recall, detected_true, predicted_true_positives, \
           predicted_false_positives, value_predicted_true_positives, \
           value_predicted_false_positives, predicted_redundant, \
           value_predicted_redundant


def precision_recall_calculator_and_detected_old(predicted_coordinates: np.array,
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
                if ((dist <= radius) and (
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
           value_undetected_predicted, redundantly_detected_predicted, \
           value_redudndantly_detected_predicted


def precision_recall_calculator_and_detected_new_old(
        predicted_coordinates: np.array,
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
                if (dist <= radius):
                    detected_true |= {tuple(true_point)}
                    detected_predicted |= {tuple(predicted_point)}
                    value_detected_predicted += [score_value]
                    flag = 'detected'
                if ((dist <= radius) and (
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
           value_undetected_predicted, redundantly_detected_predicted, \
           value_redudndantly_detected_predicted


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
