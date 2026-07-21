import numpy as np

def sample_var_std(x):

    x = np.asarray(x, dtype = np.float64)

    var = float(np.var(x, ddof = 1))
    std = float(np.sqrt(var))

    return var, std
    pass