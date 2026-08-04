import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Validate shape compatibility
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape.")

    # Validate labels belong to {-1, +1}
    unique_labels = np.unique(y_true)
    if not np.isin(unique_labels, [-1, 1]).all():
        raise ValueError("y_true labels must only contain values from {-1, +1}.")

    # Compute vectorized hinge loss: max(0, margin - y * score)
    loss = np.maximum(0.0, margin - y_true * y_score)

    # Apply reduction
    if reduction == "mean":
        return float(np.mean(loss))
    elif reduction == "sum":
        return float(np.sum(loss))
    else:
        raise ValueError("reduction must be either 'mean' or 'sum'.")
    pass