import os
import numpy as np
import shutil
import tqdm

from folder_converter import src_folder_path

# src_folder_path = "/data/user/rencw/ICL-I2PReg/dst_folder"
# dst_folder_path = "/data/user/rencw/ICL-I2PReg/dst_folder_full"
src_folder_path = "/data/user/rencw/ICL-I2PReg/dst_folder_tube"
dst_folder_path = "/data/user/rencw/ICL-I2PReg/dst_folder_tube_full"


os.makedirs(os.path.join(dst_folder_path, "depth"), exist_ok=True)
os.makedirs(os.path.join(dst_folder_path, "mask"), exist_ok=True)
os.makedirs(os.path.join(dst_folder_path, "mask_visib"), exist_ok=True)
os.makedirs(os.path.join(dst_folder_path, "npz"), exist_ok=True)
os.makedirs(os.path.join(dst_folder_path, "rgb"), exist_ok=True)

idx_max = 12

# depth
global_cnt = 0
global_scale = 20.
for each_src_folder_path in tqdm.tqdm(os.listdir(src_folder_path)):
    each_src_folder_path = os.path.join(src_folder_path, each_src_folder_path, "depth")
    for i in range(idx_max):
        each_depth_path = os.path.join(each_src_folder_path, f"{i:06d}.npy")
        if os.path.exists(each_depth_path):
            shutil.copy(each_depth_path, os.path.join(dst_folder_path, "depth", f"{global_cnt:06d}.npy"))
            # post-process
            depth = np.load(each_depth_path)
            depth = depth.astype(np.float32)
            depth = depth / global_scale
            np.save(os.path.join(dst_folder_path, "depth", f"{global_cnt:06d}.npy"), depth)
            global_cnt += 1
        else:
            print(f"Depth path not found: {each_depth_path}")
    
    

# mask
global_cnt = 0
for each_src_folder_path in tqdm.tqdm(os.listdir(src_folder_path)):
    each_src_folder_path = os.path.join(src_folder_path, each_src_folder_path, "mask")
    for i in range(idx_max):
        each_mask_path = os.path.join(each_src_folder_path, f"{i:06d}_000000.png")
        if os.path.exists(each_mask_path):
            shutil.copy(each_mask_path, os.path.join(dst_folder_path, "mask", f"{global_cnt:06d}_000000.png"))
            global_cnt += 1
        else:
            print(f"Mask path not found: {each_mask_path}")



# mask_visib
global_cnt = 0
for each_src_folder_path in tqdm.tqdm(os.listdir(src_folder_path)):
    each_src_folder_path = os.path.join(src_folder_path, each_src_folder_path, "mask_visib")
    for i in range(idx_max):
        each_mask_visib_path = os.path.join(each_src_folder_path, f"{i:06d}_000000.png")
        if os.path.exists(each_mask_visib_path):
            shutil.copy(each_mask_visib_path, os.path.join(dst_folder_path, "mask_visib", f"{global_cnt:06d}_000000.png"))
            global_cnt += 1
        else:
            print(f"Mask visib path not found: {each_mask_visib_path}")



# npz
global_cnt = 0
for each_src_folder_path in tqdm.tqdm(os.listdir(src_folder_path)):
    each_src_folder_path = os.path.join(src_folder_path, each_src_folder_path, "npz")
    for i in range(idx_max):
        each_npz_path = os.path.join(each_src_folder_path, f"{i:06d}.npz")
        if os.path.exists(each_npz_path):
            shutil.copy(each_npz_path, os.path.join(dst_folder_path, "npz", f"{global_cnt:06d}.npz"))
            # post-process
            npz_data = np.load(each_npz_path)
            pose = npz_data["obj2cam_poses"].copy()
            pose_scale_x = np.linalg.norm(pose[0, :3])
            pose_scale_y = np.linalg.norm(pose[1, :3])
            pose_scale_z = np.linalg.norm(pose[2, :3])
            pose[:3, :3] = pose[:3, :3] / np.array([pose_scale_x, pose_scale_y, pose_scale_z])
            pose[:3, 3] = pose[:3, 3] / global_scale
            new_dict = {
                "table": npz_data["table"],
                "objects": npz_data["objects"],
                "K": npz_data["K"],
                "RT": npz_data["RT"],
                "obj2cam_poses": pose,
                "azimuth": npz_data["azimuth"],
                "elevation": npz_data["elevation"],
                "distance": npz_data["distance"],
            }
            np.savez(os.path.join(dst_folder_path, "npz", f"{global_cnt:06d}.npz"), **new_dict)
            global_cnt += 1
        else:
            print(f"NPZ path not found: {each_npz_path}")


# rgb
global_cnt = 0
for each_src_folder_path in tqdm.tqdm(os.listdir(src_folder_path)):
    each_src_folder_path = os.path.join(src_folder_path, each_src_folder_path, "rgb")
    for i in range(idx_max):
        each_rgb_path = os.path.join(each_src_folder_path, f"{i:06d}.png")
        if os.path.exists(each_rgb_path):
            shutil.copy(each_rgb_path, os.path.join(dst_folder_path, "rgb", f"{global_cnt:06d}.png"))
            global_cnt += 1
        else:
            print(f"RGB path not found: {each_rgb_path}")