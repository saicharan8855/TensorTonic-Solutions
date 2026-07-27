import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):


    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Calculate error and absolute error
    error = y_true - y_pred
    abs_error = np.abs(error)
    
    # Compute quadratic loss for |e| <= delta
    quadratic = 0.5 * (error ** 2)
    
    # Compute linear loss for |e| > delta
    linear = delta * (abs_error - 0.5 * delta)
    
    # Apply piecewise condition and return mean loss
    loss = np.where(abs_error <= delta, quadratic, linear)
    return np.mean(loss)
    pass