import numpy as np

def local_response_normalization(x: np.ndarray, k: float, n: int,
                                 alpha: float, beta: float) -> np.ndarray:
    x = x.astype(np.float64)
    r = n // 2
    C = x.shape[-1]
    
    # Compute x^2 elementwise
    x_sq = x ** 2
    
    # Initialize output array
    y = np.zeros_like(x)
    
    # Loop across the channel dimension
    for c in range(C):
        start = max(0, c - r)
        end = min(C, c + r + 1)
        
        # Sum squared values across the channel neighborhood window
        sum_sq = np.sum(x_sq[..., start:end], axis=-1)
        
        # Apply LRN formula
        denom = (k + (alpha / n) * sum_sq) ** beta
        y[..., c] = x[..., c] / denom
        
    return y
    """
    Returns the float64 channel-normalized tensor.
    """
    pass