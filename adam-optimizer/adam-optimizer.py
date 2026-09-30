import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    param = np.asarray(param)
    grad = np.asarray(grad)
    m = np.asarray(m)
    v = np.asarray(v)

    mt = beta1 * m + (1 - beta1)*grad 
    vt = beta2 * v + (1 - beta2)*(grad*grad)
    m_cap = mt /  (1 - (beta1)**t)
    v_cap = vt / (1 - (beta2)**t)
    teta = param - lr*m_cap/(np.sqrt(v_cap) + eps)

    return teta, mt, vt
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    # Write code here
    pass