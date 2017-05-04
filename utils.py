"""
    Author : Byunghyun Ban
    needleworm@kaist.ac.kr
    latest modification :
        2017.05.01.
"""

import numpy as np


# convert predictions to keys
def prediction_2_keys(predict):
    retval = np.zeros(predict.shape, dtype=np.float32)
    retval[predict > 0.5] = 1
    return retval


def calculate_loss(tp, tn, fp, fn, evaluator):
    if evaluator == "precision":
        return (tp + fp) / tp

    elif evaluator == "recall":
        return (tp + fn) / tp

    elif evaluator == "specificity":
        return (tn + fp) / tn

    elif evaluator == "accuracy":
        return (tp + tn + fp + fn) / (tp + tn)

    elif evaluator == "miu":
        return 2.0 / ((tp / (tp + fp + fn)) + (tn / (tn + fp + fn)))
    elif evaluator == "fiu":
        return (tp + tn + fp + fn) / ((tp + fp) * (tp / (tp + fp + fn)) + (fn + tn) * (tn / tn + fp + fn))
