import numpy as np

def alexnet_conv1(image: np.ndarray, kernel: np.ndarray, bias: np.ndarray,
                   stride: int, padding: int) -> np.ndarray:

    image = np.asarray(image, dtype = np.float64)
    kernel = np.asarray(kernel, dtype = np.float64)
    bias = np.asarray(bias, dtype = np.float64)

    B, H, W, Cin = image.shape
    Kh, Kw, Cin_K, Cout = kernel.shape

    assert Cin == Cin_K
    if padding > 0:
        image_padded = np.pad(
            image,
            pad_width=((0, 0), (padding, padding), (padding, padding), (0, 0)),
            mode='constant',
            constant_values=0.0
        )
    else:
        image_padded = image


    H_pad , W_pad = image_padded.shape[1], image_padded.shape[2]

    Hout = (H + 2 * padding - Kh) // stride + 1
    Wout = (W + 2 * padding - Kw) // stride + 1

    Y = np.zeros((B, Hout, Wout, Cout), dtype=np.float64)

    for i in range(Hout):
        for j in range(Wout):
            # Extract spatial slice for all batch items and input channels
            # Slice shape: (B, Kh, Kw, Cin)
            h_start = i * stride
            w_start = j * stride
            patch = image_padded[:, h_start:h_start + Kh, w_start:w_start + Kw, :]
            
            # Multiply patch elementwise with kernel and sum over Kh, Kw, Cin
            # patch: (B, Kh, Kw, Cin, 1) via newaxis
            # kernel: (Kh, Kw, Cin, Cout)
            # Result after sum over axes (1, 2, 3) gives shape (B, Cout)
            Y[:, i, j, :] = np.sum(patch[:, :, :, :, np.newaxis] * kernel, axis=(1, 2, 3)) + bias
            
    return Y
    """
    Returns the float64 NHWC convolution output.
    """
    pass