import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x, dtype = np.float64)
    """
    Returns the elementwise float64 ReLU output.
    """
    pass