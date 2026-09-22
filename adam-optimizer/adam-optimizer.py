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

    m_t = beta1 * m + (1 - beta1) * grad
    v_t = beta2 * v + (1 - beta2) * (grad**2)
    beta1_power = beta1 ** t
    beta2_power = beta2 ** t
    m_cap = m_t / (1 - beta1_power)
    v_cap = v_t / (1 - beta2_power)
    new_param = param - lr*m_cap/(np.sqrt(v_cap) + eps)

    return new_param, m_t, v_t
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    # Write code here
    pass