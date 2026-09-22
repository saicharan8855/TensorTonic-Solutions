import numpy as np

def vgg_conv_block(x: np.ndarray, kernels: list,
                   biases: list) -> np.ndarray:

    out = x.astype(np.float64)

    for K, b in zip(kernels, biases):
        K = np.asarray(K, dtype = np.float64)
        b = np.asarray(b, dtype = np.float64)

        Kh, Kw, C_in, C_out = K.shape
        pad_h = Kh // 2
        pad_w = Kw // 2

        padded = np.pad(out, ((0,0), (pad_h, pad_h), (pad_w, pad_w), (0,0)), mode = 'constant')


        N, H, W, _ = out.shape
        conv_out = np.zeros((N, H, W, C_out), dtype = np.float64)

        for i in range(H):
            for j in range(W):
                # Extract patch: (N, Kh, Kw, C_in)
                patch = padded[:, i:i+Kh, j:j+Kw, :]
                # Sum over Kh, Kw, C_in (axes 1, 2, 3 of patch and 0, 1, 2 of K)
                conv_out[:, i, j, :] = np.tensordot(patch, K, axes=([1, 2, 3], [0, 1, 2]))


        out = np.maximum(0, conv_out + b)
    return out

    
    """
    Returns the float64 block activations in NHWC layout.
    """
    pass