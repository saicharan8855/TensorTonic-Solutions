import numpy as np

def vector_norm_3d(v):

    v = np.asarray(v, dtype=float)
    
    # Calculate Euclidean norm across the last axis (-1 handles both 1D and 2D arrays)
    norm = np.linalg.norm(v, axis=-1)
    
    # Convert back to a python float scalar if the input is a single vector
    if v.ndim == 1:
        return float(norm)
    
    return norm
    pass