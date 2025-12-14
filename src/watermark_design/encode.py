from __future__ import annotations

import numpy as np


def encode_to_bitstream(watermark: np.ndarray) -> np.ndarray:
    """
    Convert a 2D binary watermark into a 1D bitstream using row-major order.

    Constraints:
    - Output length MUST equal H*W
    - Output values MUST be {0,1}
    - No embedding/extraction/evaluation and no file I/O

    Args:
        watermark: 2D NumPy array (H, W), expected values in {0,1}

    Returns:
        bitstream: 1D NumPy array (H*W,), dtype=uint8, values in {0,1}
    """
    if not isinstance(watermark, np.ndarray):
        raise TypeError("watermark must be a NumPy array.")
    if watermark.ndim != 2:
        raise ValueError(f"watermark must be 2D, got shape {watermark.shape}")

    # Enforce strict binary {0,1}
    wm01 = (watermark > 0).astype(np.uint8)

    # Row-major flatten (C-order)
    bitstream = wm01.reshape(-1).astype(np.uint8)

    # Integration constraint: length must be H*W
    h, w = wm01.shape
    expected = h * w
    if bitstream.size != expected:
        raise RuntimeError(f"Bitstream length mismatch: {bitstream.size} != {expected}")

    # Ensure binary
    uniq = set(np.unique(bitstream).tolist())
    if not uniq.issubset({0, 1}):
        raise RuntimeError(f"Bitstream contains non-binary values: {uniq}")

    return bitstream


def _decode_from_bitstream(bitstream: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    Helper used ONLY for the self-test: reshape bitstream back to 2D watermark.
    """
    if bitstream.ndim != 1:
        raise ValueError("bitstream must be 1D for decoding.")
    h, w = int(shape[0]), int(shape[1])
    if bitstream.size != h * w:
        raise ValueError("bitstream length does not match target shape.")
    return (bitstream.astype(np.uint8) & 1).reshape(h, w)


if __name__ == "__main__":
    # Self-test: encode then decode and verify equality
    from watermark_design.design import design_watermark

    wm = design_watermark((32, 32))
    bs = encode_to_bitstream(wm)
    wm2 = _decode_from_bitstream(bs, wm.shape)

    assert bs.size == wm.size, "Bitstream length != H*W"
    assert np.array_equal(wm, wm2), "Decoded watermark does not match original watermark"
    print("[encode.py] Self-test passed.")
