import numpy as np

def random_crop(image: np.ndarray, crop_size: int,
                crop_y: int, crop_x: int) -> np.ndarray:
    crop = image[crop_y : crop_y + crop_size, crop_x : crop_x + crop_size, :]
    return crop.astype(np.float64).copy()
    """
    Returns the float64 crop at the supplied coordinates.
    """
    pass

def random_horizontal_flip(image: np.ndarray, p: float,
                           flip_rand: float) -> np.ndarray:
    if flip_rand < p:
        # Reverse along width axis (axis 1 in HWC layout)
        flipped = image[:, ::-1, :]
        return flipped.astype(np.float64).copy()
    
    return image.astype(np.float64).copy()
    """
    Returns a new image, flipped when flip_rand is less than p.
    """
    pass