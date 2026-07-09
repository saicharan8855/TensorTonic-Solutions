import numpy as np

def dot_product(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError("mismatched length")
    return np.dot(x, y)
    pass