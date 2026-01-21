import os
import pickle
import numpy as np
import OpenEXR
import Imath

def read_exr(exr_path):
    exr_file = OpenEXR.InputFile(exr_path)
    
    channels = list(exr_file.header()['channels'].keys())
    print("Channels in EXR:", channels)
    
    # 获取图像尺寸
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # 设置通道类型
    pt = Imath.PixelType(Imath.PixelType.FLOAT)  # 32-bit float
    
    # 读取 RGB 通道
    r = np.frombuffer(exr_file.channel('R', pt), dtype=np.float32).reshape(height, width)
    g = np.frombuffer(exr_file.channel('G', pt), dtype=np.float32).reshape(height, width)
    b = np.frombuffer(exr_file.channel('B', pt), dtype=np.float32).reshape(height, width)
    a = np.frombuffer(exr_file.channel('A', pt), dtype=np.float32).reshape(height, width)
    
    # 合并为 HWC 数组
    img = np.stack([r, g, b], axis=2)  # float32, HWC
    alpha = a
    
    return img, alpha


file_path = "/data/user/rencw/ICL-I2PReg/views/plane_gray__test_tube_rack"

for file in os.listdir(file_path):
    if file.endswith(".exr"):
        exr_path = os.path.join(file_path, file)
        img, alpha = read_exr(exr_path)
        print(img.shape, alpha.shape, img.max(), alpha.max(), img.min(), alpha.min(), exr_path)
        import pdb; pdb.set_trace()
        # alpha = 1: img, alpha = 0
        
        
        