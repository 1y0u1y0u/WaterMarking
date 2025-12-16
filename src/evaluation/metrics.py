import numpy as np
import math

def calculate_psnr(original: np.ndarray, modified: np.ndarray) -> float:
    """
    計算峰值訊噪比 (PSNR) 以評估影像品質 [cite: 166, 172]。
    通常用於衡量浮水印嵌入後的不可見性。
    
    Args:
        original: 原始影像 (Cover Image)
        modified: 修改後的影像 (Stego Image or Attacked Image)
        
    Returns:
        PSNR 值 (float)，單位為 dB。數值越大代表差異越小。
    """
    # 檢查輸入影像是否可比較 [cite: 176]
    if original.shape != modified.shape:
        raise ValueError("Input images must have the same dimensions")
    
    # 計算均方誤差 (MSE)
    # 需轉為 float 以避免溢位
    mse = np.mean((original.astype(np.float64) - modified.astype(np.float64)) ** 2)
    
    # 如果兩張圖完全一樣，MSE 為 0，PSNR 趨近無限大
    if mse == 0:
        return float('inf')
    
    # 影像像素最大值通常為 255
    max_pixel = 255.0
    
    # PSNR 公式: 10 * log10(MAX^2 / MSE)
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    
    return psnr

def calculate_ber(original_bits: np.ndarray, extracted_bits: np.ndarray) -> float:
    """
    計算位元錯誤率 (Bit Error Rate, BER) [cite: 167, 172]。
    用於評估浮水印在攻擊後的恢復品質。
    
    Args:
        original_bits: 原始浮水印位元流 (1D numpy array, 0/1)
        extracted_bits: 提取出的浮水印位元流 (1D numpy array, 0/1)
        
    Returns:
        BER 值 (float, 0.0 ~ 1.0)。0.0 代表完全正確，數值越小越好。
    """
    # 確保輸入是平面化的 1D 陣列
    orig = original_bits.flatten()
    extr = extracted_bits.flatten()
    
    # 檢查長度是否一致 [cite: 177]
    if len(orig) != len(extr):
        raise ValueError(f"Bitstreams must be of equal length. Original: {len(orig)}, Extracted: {len(extr)}")
    
    # 計算錯誤的位元總數
    # 由於是 0/1 二值，可以直接算不相等的數量
    errors = np.sum(orig != extr)
    
    # BER = 錯誤位元數 / 總位元數
    ber = errors / len(orig)
    
    return ber