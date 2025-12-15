import numpy as np
# from typing import Tuple # decode.py 已經用到，這邊可以省略

def extract_bitstream(
    stego: np.ndarray, 
    bit_plane: int, 
    channel: int, 
    length: int
) -> np.ndarray:
    """
    從 stego 圖像中提取「嵌入後的位元流」。
    
    Args:
        stego (np.ndarray): 嵌入浮水印後的圖像 (BGR, uint8)。
        bit_plane (int): 要提取的位元平面索引。
        channel (int): 要提取的顏色通道 (0=B, 1=G, 2=R)。
        length (int): 要提取的總位元數。
        
    Returns:
        np.ndarray: 提取出的 1D 位元流，即 embedded_bits (0/1)。
    """
    # 1. 取得指定通道的像素值 (BGR 格式)
    stego_channel = stego[:, :, channel]

    # 2. 提取 'bit_plane' 上的位元，這就是嵌入方寫入的 embedded_bit
    # 提取後的位元 (embedded_bit)
    embedded_bit = (stego_channel >> bit_plane) & 1
    
    # 3. 根據行優先 (row-major) 順序，將 2D 陣列攤平為 1D 位元流
    bitstream = embedded_bit.flatten()

    # 4. 根據所需長度截取
    if length > len(bitstream):
        length = len(bitstream) 
        
    return bitstream[:length].astype(np.uint8)