import numpy as np

def copy_image(image: np.ndarray) -> np.ndarray:
    """
    Return a deep copy of the image to avoid modifying the original.
    """
    return np.copy(image)

