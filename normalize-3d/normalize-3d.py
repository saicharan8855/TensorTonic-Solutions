import numpy as np

def normalize_3d(v):

    v = np.asarray(v, dtype=float)
    
    # Calculate norms along the last axis, keeping dimensions for broadcasting
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    
    # Avoid division by zero: divide where norm > 1e-10, else return original (zeros)
    return np.where(norms > 1e-10, v / norms, v)
    pass