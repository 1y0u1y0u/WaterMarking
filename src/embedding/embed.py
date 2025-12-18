import numpy as np
from .image_utils import copy_image


def embed_bit(pixel: int, bit: int, bit_plane: int) -> int:
    pixel = np.uint8(pixel)
    mask = np.uint8(1 << bit_plane)

    original_bit = (pixel >> bit_plane) & 1
    embedded_bit = original_bit ^ bit

    pixel &= np.uint8(0xFF ^ mask)
    pixel |= np.uint8(embedded_bit << bit_plane)

    return pixel


def embed_watermark(
    image: np.ndarray,
    bitstream: np.ndarray,
    bit_plane: int,
    channel: int
) -> np.ndarray:
    stego = copy_image(image)
    h, w = image.shape[:2]
    flat_index = 0

    for i in range(h):
        for j in range(w):
            if flat_index >= bitstream.size:
                break

            stego[i, j, channel] = embed_bit(
                stego[i, j, channel],
                int(bitstream[flat_index]),
                bit_plane
            )
            flat_index += 1

    return stego


