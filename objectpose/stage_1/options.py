import numpy as np
import math
import torch


class Options:
    def __init__(self):
        self.is_fine_resolution = True
        self.is_remove_ground = False
        self.translation_max = 10.0
        self.test_translation_max = 10.0

        self.is_centered = False
        self.crop_original_top_rows = 100
        self.img_scale = 0.2
        self.img_H = 160  # after scale before crop 800 * 0.4 = 320
        self.img_W = 320  # after scale before crop 1600 * 0.4 = 640
        self.as_gray = False
        # the fine resolution is img_H/scale x img_W/scale
        self.img_fine_resolution_scale = 32

        self.num_kpt=512
        
        self.input_pt_num = 40960
        self.node_a_num = 256
        self.node_b_num = 256
        self.k_ab = 32
        self.k_interp_ab = 3
        self.k_interp_point_a = 3
        self.k_interp_point_b = 3

        self.P_tx_amplitude = 10
        self.P_ty_amplitude = 0
        self.P_tz_amplitude = 10
        self.P_Rx_amplitude = 0
        self.P_Ry_amplitude = 2.0 * math.pi
        self.P_Rz_amplitude = 0

