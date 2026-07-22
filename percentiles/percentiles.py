import numpy as np

def percentiles(x, q):

    return np.percentile(x, q, method = 'linear')
    pass