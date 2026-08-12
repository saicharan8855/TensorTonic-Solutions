import numpy as np

def r2_score(y_true, y_pred) -> float:


    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Handle the constant-target edge case
    if np.all(y_true == y_true[0]):
        return 1.0 if np.array_equal(y_true, y_pred) else 0.0
    
    # Residual Sum of Squares (SSE / SS_res)
    sse = np.sum((y_true - y_pred) ** 2)
    
    # Total Sum of Squares (SST / SS_tot)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    
    return float(1.0 - (sse / sst))
    pass