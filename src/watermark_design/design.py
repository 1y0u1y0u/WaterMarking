
from __future__ import annotations

from typing import Tuple
import numpy as np


def design_watermark(size: Tuple[int, int] = (32, 32)) -> np.ndarray:
    """
    Generate a deterministic binary watermark (values in {0,1}) as a 2D NumPy array.

    Design rationale:
    - Deterministic across devices/runs (fixed seed derived from size).
    - Includes high-contrast structural patterns (border + diagonals + finder blocks)
      so the extracted watermark remains recognizable even with mild corruption.

    Args:
        size: (height, width)

    Returns:
        watermark: np.ndarray of shape (H, W), dtype=uint8, values in {0,1}
    """
    if not (isinstance(size, tuple) and len(size) == 2):
        raise TypeError("size must be a tuple of (height, width).")
    h, w = int(size[0]), int(size[1])
    if h <= 0 or w <= 0:
        raise ValueError("Watermark size must be positive.")

    # Base: deterministic pseudo-random field (helps uniqueness / non-triviality)
    # Seed derived from size to keep it reproducible without external state.
    seed = (h * 1000003 + w * 9176) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    wm = (rng.integers(0, 2, size=(h, w), dtype=np.uint8) & 1).astype(np.uint8)

    # Add a 1-pixel border (improves visual recognizability after extraction)
    wm[0, :] = 1
    wm[-1, :] = 1
    wm[:, 0] = 1
    wm[:, -1] = 1

    # Add diagonals (structure)
    d = min(h, w)
    wm[np.arange(d), np.arange(d)] = 1
    wm[np.arange(d), (w - 1) - np.arange(d)] = 1

    # Add "finder" blocks (top-left, top-right, bottom-left) if size allows
    # These are common in watermark/marker designs to aid recognition.
    block = max(3, min(h, w) // 6)  # scales with size, at least 3
    block = min(block, h, w)
    wm[1:1 + block, 1:1 + block] = 1
    wm[1:1 + block, w - 1 - block:w - 1] = 1
    wm[h - 1 - block:h - 1, 1:1 + block] = 1

    # Ensure strict binary {0,1}
    wm = (wm > 0).astype(np.uint8)
    return wm


if __name__ == "__main__":
    # Minimal self-check for this module alone (no file I/O).
    wm = design_watermark((32, 32))
    assert wm.shape == (32, 32), f"Unexpected shape: {wm.shape}"
    uniq = set(np.unique(wm).tolist())
    assert uniq.issubset({0, 1}), f"Non-binary values found: {uniq}"
    print("[design.py] Self-test passed.")
