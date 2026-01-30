from PIL import Image
import numpy as np
import os

def rendering2alpha(rendering_path, alpha_path):
    rendering = Image.open(rendering_path)
    rendering = np.array(rendering)
    alpha = rendering[:, :, 3]
    alpha = alpha.astype(np.uint8)
    alpha = Image.fromarray(alpha)
    alpha.save(alpha_path)
    
def alpha_compare(alpha_path1, alpha_path2, diff_path):
    alpha1 = Image.open(alpha_path1)
    alpha2 = Image.open(alpha_path2)
    alpha1 = np.array(alpha1).astype(np.float32)
    alpha2 = np.array(alpha2).astype(np.float32)
    diff = alpha1 - alpha2
    diff = np.abs(diff)
    diff = diff.astype(np.uint8)
    diff = Image.fromarray(diff)
    diff.save(diff_path)

if __name__ == "__main__":
    rendering_path = "/data/user/rencw/ICL-I2PReg/render.png"
    alpha_path = "/data/user/rencw/ICL-I2PReg/rendering_alpha.png"
    rendering2alpha(rendering_path, alpha_path)
    
    diff_path = "/data/user/rencw/ICL-I2PReg/rendering_alpha_diff.png"
    alpha_path1 = "/data/user/rencw/ICL-I2PReg/rendering_alpha.png"
    alpha_path2 = "/data/user/rencw/ICL-I2PReg/views_tube/plane_table.glb__tube_thick.blend/002_mask.png"
    
    alpha_compare(alpha_path1, alpha_path2, diff_path)