import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    \
    if matrix is None:
        return None

    try:
        arr = np.array(matrix, dtype = float)

    except (ValueError, TypeError):
        return None

    if arr.ndim != 2 or axis not in [0, 1, None] or norm_type not in ['l1', 'l2', 'max']:
        return None

    # 2. Compute the norm based on the norm_type
    if norm_type == 'l1':
        # Manhattan norm: sum of absolute values
        norm = np.sum(np.abs(arr), axis=axis, keepdims=True)
    elif norm_type == 'l2':
        # Euclidean norm: square root of sum of squares
        norm = np.sqrt(np.sum(np.square(arr), axis=axis, keepdims=True))
    elif norm_type == 'max':
        # Infinity norm: maximum absolute value
        norm = np.max(np.abs(arr), axis=axis, keepdims=True)
    else:
        return None

    normalized_arr = np.where(norm == 0, 0.0, arr / norm)
    
    return normalized_arr
        
    pass