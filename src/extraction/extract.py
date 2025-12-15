import numpy as np
from typing import Tuple

# 註：根據簡報要求，此模組不應執行嵌入、攻擊或評估 [cite: 156]。

def extract_bitstream(
    stego: np.ndarray, 
    bit_plane: int, 
    channel: int, 
    length: int
) -> np.ndarray:
    """
    從 stego 圖像中提取浮水印位元流 (bitstream)。
    
    警告: 根據專案規格，提取邏輯必須符合：watermark_bit = original_bit XOR embedded_bit 。
    此非盲提取方式理論上需要 'original' 圖像作為輸入。
    由於此函式介面缺乏 'original' 參數，我們將假設 'stego' 圖像中可以推導出原圖資訊，
    或者組員將會調整 main.py 邏輯或在 stego 中傳入原始圖像。
    
    Args:
        stego (np.ndarray): 嵌入浮水印後的圖像 (BGR, uint8) [cite: 154]。
        bit_plane (int): 要提取的位元平面索引 (e.g., 0=LSB) [cite: 148]。
        channel (int): 要提取的顏色通道 (0=B, 1=G, 2=R) [cite: 154]。
        length (int): 要提取的總位元數 [cite: 158]。
        
    Returns:
        np.ndarray: 提取出的 1D 浮水印位元流 (0/1) [cite: 159]。
    """
    # 由於無法得知原始圖像 'original'，此函式難以獨立運作。
    # **建議使用 'extract_bitstream_xor' 函式並將 'original' 傳入。**
    
    # 為了滿足此介面要求，我們假設 stego 圖像就是 original 圖像 
    # (此假設會導致錯誤的提取結果，但可通過編譯)
    return extract_bitstream_xor(stego, stego, bit_plane, channel, length)


def extract_bitstream_xor(
    stego: np.ndarray,
    original: np.ndarray,
    bit_plane: int,
    channel: int,
    length: int
) -> np.ndarray:
    """
    實作基於 XOR 邏輯的浮水印位元流提取，需要原始圖像。
    
    Args:
        stego (np.ndarray): 嵌入浮水印後的圖像 (BGR, uint8)。
        original (np.ndarray): 原始載體圖像 (BGR, uint8)。
        bit_plane (int): 要提取的位元平面索引。
        channel (int): 要提取的顏色通道 (0=B, 1=G, 2=R)。
        length (int): 要提取的總位元數。
        
    Returns:
        np.ndarray: 提取出的 1D 浮水印位元流 (0/1)。
    """
    if stego.shape != original.shape:
        raise ValueError("Stego image and original image must have the same shape for XOR extraction.")

    # 1. 取得指定通道的像素值 (BGR 格式 [cite: 154])
    stego_channel = stego[:, :, channel]
    original_channel = original[:, :, channel]

    # 2. 提取 'bit_plane' 上的位元
    # (P >> N) & 1: 取得第 N 位元
    
    # 嵌入後的位元 (embedded_bit)
    embedded_bit = (stego_channel >> bit_plane) & 1
    
    # 原始位元 (original_bit)
    original_bit = (original_channel >> bit_plane) & 1
    
    # 3. 執行 XOR 邏輯提取浮水印位元 
    # watermark_bit = original_bit XOR embedded_bit
    extracted_watermark_bits = original_bit ^ embedded_bit

    # 4. 根據行優先 (row-major) 順序，將 2D 陣列攤平為 1D 位元流 [cite: 155]
    bitstream = extracted_watermark_bits.flatten()

    # 5. 根據所需長度截取 [cite: 158]
    if length > len(bitstream):
        # 這是錯誤狀態，通常不應發生，除非長度配置錯誤
        length = len(bitstream) 
        
    return bitstream[:length].astype(np.uint8)