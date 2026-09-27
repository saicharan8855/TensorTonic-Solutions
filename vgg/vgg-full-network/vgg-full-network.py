import numpy as np

def vgg_forward(x: np.ndarray, config: list, kernels: list, biases: list, classifier: dict) -> np.ndarray:
    """
    Returns float64 class logits with shape (B, C_classes).
    """
    x = np.array(x, dtype=np.float64)
    kernel_idx = 0
    
    # 1. Feature Extractor (Convolution & Max Pooling)
    for entry in config:
        if entry == "M":
            # Max pooling (2x2 with stride 2 across spatial H and W dimensions)
            B, H, W, C = x.shape
            x = x.reshape(B, H // 2, 2, W // 2, 2, C).max(axis=(2, 4))
        else:
            # 2D Convolution (NHWC format)
            K = np.array(kernels[kernel_idx], dtype=np.float64)  # Shape: (kH, kW, C_in, C_out)
            b = np.array(biases[kernel_idx], dtype=np.float64)    # Shape: (C_out,)
            kernel_idx += 1
            
            kH, kW, C_in, C_out = K.shape
            
            # Apply padding (same padding for kH x kW kernels, typically pad = kH // 2)
            pad_h = kH // 2
            pad_w = kW // 2
            
            if pad_h > 0 or pad_w > 0:
                x_padded = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
            else:
                x_padded = x
                
            B, H_pad, W_pad, _ = x_padded.shape
            out_H = H_pad - kH + 1
            out_W = W_pad - kW + 1
            
            out = np.zeros((B, out_H, out_W, C_out), dtype=np.float64)
            for i in range(out_H):
                for j in range(out_W):
                    patch = x_padded[:, i:i+kH, j:j+kW, :]  # Shape: (B, kH, kW, C_in)
                    out[:, i, j, :] = np.tensordot(patch, K, axes=([1, 2, 3], [0, 1, 2])) + b
            
            # Apply ReLU activation
            x = np.maximum(0, out)
            
    # 2. Flatten spatial features into 2D matrix (B, Flattened_Features)
    B = x.shape[0]
    x = x.reshape(B, -1)
    
    # 3. Classifier Head (Affine + ReLU for first 2 layers, Affine only for final layer)
    W1, b1 = np.array(classifier["W1"], dtype=np.float64), np.array(classifier["b1"], dtype=np.float64)
    W2, b2 = np.array(classifier["W2"], dtype=np.float64), np.array(classifier["b2"], dtype=np.float64)
    W3, b3 = np.array(classifier["W3"], dtype=np.float64), np.array(classifier["b3"], dtype=np.float64)
    
    # Layer 1: Affine + ReLU
    x = np.maximum(0, np.dot(x, W1) + b1)
    
    # Layer 2: Affine + ReLU
    x = np.maximum(0, np.dot(x, W2) + b2)
    
    # Layer 3: Final Affine projection (Raw Logits, no activation)
    logits = np.dot(x, W3) + b3
    
    return logits