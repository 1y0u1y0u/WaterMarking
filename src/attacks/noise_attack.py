import numpy as np

def simulate_noise_attack(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    模擬高斯雜訊攻擊 [cite: 165, 171]。
    
    Args:
        image: 輸入影像 (BGR format, numpy array)
        sigma: 高斯分佈的標準差，數值越大雜訊越強
        
    Returns:
        加入雜訊後的影像 (uint8, 0-255)
    """
    # 產生與原圖形狀相同的高斯雜訊
    # mean=0, standard deviation=sigma
    noise = np.random.normal(0, sigma, image.shape)
    
    # 將影像轉為 float 進行加法運算，避免溢位
    noisy_image = image.astype(np.float32) + noise
    
    # 將數值限制在 [0, 255] 區間內 (Clipping)
    noisy_image = np.clip(noisy_image, 0, 255)
    
    # 轉回 uint8 格式回傳
    return noisy_image.astype(np.uint8)