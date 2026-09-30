import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    x = np.asarray(x)
    return 1 / (1 + np.exp(-x))
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    # Write code here
    pass