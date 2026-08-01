import numpy as np

def apply_homogeneous_transform(T, points):

    T = np.asarray(T)
    pts = np.asarray(points)
    
    is_single = pts.ndim == 1
    
    # Reshape single point (3,) to batch (1, 3) for uniform processing
    if is_single:
        pts = pts[np.newaxis, :]
        
    # Convert to homogeneous coordinates by appending 1s: shape (N, 4)
    ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    pts_h = np.hstack([pts, ones])
    
    # Matrix multiplication: p_h' = T @ p_h  =>  transformed_h = pts_h @ T.T
    transformed_h = pts_h @ T.T
    
    # Extract spatial coordinates
    result = transformed_h[:, :3]
    
    return result[0] if is_single else result
    pass