import numpy as np

def rotate_around_z(points, theta):

    pts = np.asarray(points)
    original_shape = pts.shape
    
    # Reshape to (N, 3) for unified matrix multiplication
    pts_2d = pts.reshape(-1, 3)
    
    # Construct the 3D Z-axis rotation matrix
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    R_z = np.array([
        [cos_t, -sin_t, 0],
        [sin_t,  cos_t, 0],
        [    0,      0, 1]
    ])
    
    # Perform matrix multiplication: (N, 3) @ (3, 3).T = (N, 3)
    rotated = pts_2d @ R_z.T
    
    # Reshape back to original shape (3,) or (N, 3)
    return rotated.reshape(original_shape)
    pass