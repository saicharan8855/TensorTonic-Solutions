import numpy as np

def conv2d_same(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Performs a same-padded 2D convolution in NHWC layout."""
    N, H, W, C_in = x.shape
    k_h, k_w, _, C_out = kernel.shape
    
    # Calculate SAME padding
    pad_h = (k_h - 1) // 2
    pad_w = (k_w - 1) // 2
    
    # Pad input along spatial dimensions (H, W)
    x_padded = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
    
    out = np.zeros((N, H, W, C_out), dtype=np.float64)
    
    for i in range(H):
        for j in range(W):
            # Extract spatial patch: shape (N, k_h, k_w, C_in)
            patch = x_padded[:, i:i+k_h, j:j+k_w, :]
            # Multiply with kernel: einsum 'nhwc,hwco->no' sums over h, w, c
            out[:, i, j, :] = np.einsum('nhwc,hwco->no', patch, kernel) + bias
            
    return out

def max_pool_2x2(x: np.ndarray) -> np.ndarray:
    """Performs 2x2 non-overlapping max pooling in NHWC layout."""
    N, H, W, C = x.shape
    # Reshape and take max across 2x2 local regions
    out = x.reshape(N, H // 2, 2, W // 2, 2, C)
    return out.max(axis=(2, 4))

def vgg_features(x: np.ndarray, config: list, kernels: list, biases: list) -> np.ndarray:
    """Returns the float64 VGG features in NHWC layout."""
    out = x.astype(np.float64)
    kernel_idx = 0
    
    for layer in config:
        if layer == "M":
            out = max_pool_2x2(out)
        else:
            k = np.array(kernels[kernel_idx], dtype=np.float64)
            b = np.array(biases[kernel_idx], dtype=np.float64)
            
            # Convolution + Bias
            out = conv2d_same(out, k, b)
            # ReLU activation
            out = np.maximum(0, out)
            
            kernel_idx += 1
            
    return out

    
    """
    Returns the float64 VGG features in NHWC layout.
    """
    pass