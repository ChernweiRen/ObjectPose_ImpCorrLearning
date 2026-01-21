import os
import shutil

folder_path = "/data/user/rencw/ICL-I2PReg/views"

for each_folder_path in os.listdir(folder_path):
    each_folder_path = os.path.join(folder_path, each_folder_path)
    for each_file in os.listdir(each_folder_path):
        if each_file.endswith(".npz"):
            # if has .png, then maintain. otherwise, delete
            if os.path.exists(os.path.join(each_folder_path, each_file.replace(".npz", ".png"))):
                continue
            else:
                print(f"Delete {os.path.join(each_folder_path, each_file)}")