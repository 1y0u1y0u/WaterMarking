import cv2
import numpy as np

def simulate_jpeg_attack(image: np.ndarray, quality: int) -> np.ndarray:
    """
    模擬 JPEG 壓縮攻擊 [cite: 164, 171]。
    
    Args:
        image: 輸入影像 (BGR format, numpy array)
        quality: JPEG 品質參數 (0-100)，數值越低壓縮越嚴重
        
    Returns:
        壓縮後再解碼的影像 (保持原尺寸與形狀 [cite: 175])
    """
    # 參數檢查：確保品質在合理範圍
    quality = max(0, min(quality, 100))
    
    # 使用 OpenCV 進行記憶體內的編碼與解碼
    # cv2.imencode 將影像編碼為 JPEG 格式的 byte stream
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encoded_img = cv2.imencode('.jpg', image, encode_param)
    
    if not result:
        raise ValueError("JPEG encoding failed")
        
    # cv2.imdecode 將 byte stream 解碼回影像陣列
    # 這過程會產生有損壓縮的失真
    decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    
    return decoded_img