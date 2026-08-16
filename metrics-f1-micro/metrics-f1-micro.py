import numpy as np
def f1_micro(y_true, y_pred) -> float:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if len(y_true_arr) == 0:
        return 0.0

    # For single-label multi-class classification:
    # TP is the number of correct predictions.
    # Every incorrect prediction contributes exactly 1 to FP and 1 to FN.
    tp = int(np.sum(y_true_arr == y_pred_arr))
    fp = len(y_true_arr) - tp
    fn = fp

    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return 0.0

    return float((2 * tp) / denominator)
    pass