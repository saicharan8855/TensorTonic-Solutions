import numpy as np

def geometric_pmf_mean(k, p):

    k_arr = np.asarray(k)

    pmf = ((1 - p) ** (k_arr - 1)) * p 

    mean = float(1/p)

    return pmf, mean
    pass