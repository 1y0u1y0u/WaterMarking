import numpy as np
from typing import Tuple

# 註：根據簡報要求，此模組不應執行嵌入、攻擊或評估 [cite: 156]。

def decode_to_image(bitstream: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    將 1D 浮水印位元流還原為 2D 浮水印圖像。

    Args:
        bitstream (np.ndarray): 1D 浮水印位元流 (0/1)。
        shape (tuple): 浮水印圖像的 (height, width) [cite: 158]。

    Returns:
        np.ndarray: 2D 浮水印圖像，值為 (0, 1) [cite: 159]。
    """
    H, W = shape
    if bitstream.size != H * W:
        raise ValueError(f"Bitstream size ({bitstream.size}) does not match target shape size ({H * W}).")
        
    # 直接使用 reshape 函式，因為嵌入/提取都遵循行優先 (row-major) [cite: 155]
    watermark_image = bitstream.reshape(H, W)
    
    # 確保輸出為二值 (0/1) [cite: 159]
    return watermark_image.astype(np.uint8)


# --- 獨立性測試 (Self-Test) ---

if __name__ == "__main__":
    print("--- 浮水印解碼模組自測 (Self-Test) ---")
    
    # 模擬 1D 位元流 (2x3 圖像)
    test_bitstream = np.array([1, 0, 0, 1, 1, 0], dtype=np.uint8)
    target_shape = (2, 3) # (Height, Width)
    
    decoded_image = decode_to_image(test_bitstream, target_shape)

    # 預期結果
    expected_image = np.array([
        [1, 0, 0],
        [1, 1, 0]
    ], dtype=np.uint8)

    print(f"原始位元流: {test_bitstream}")
    print(f"預期解碼結果 ({target_shape}):\n{expected_image}")
    print(f"實際解碼結果:\n{decoded_image}")

    match = np.array_equal(expected_image, decoded_image)
    print(f"\n解碼圖像是否正確: {'是 (Success!)' if match else '否 (Failure)'}")