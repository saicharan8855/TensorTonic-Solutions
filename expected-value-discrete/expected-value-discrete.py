import numpy as np

def expected_value_discrete(x, p):

    x = np.asarray(x)
    p = np.asarray(p) 

    if x.shape != p.shape:
        raise ValueError("..")

    if not np.allclose(np.sum(p), 1.0, atol = 1e-6):
        raise ValueError("..")

    ev = np.sum(x*p)
    return float(ev)
    pass
