import bpy
import sys
import os

# 填入你刚才保存的那个新文件的路径，或者是你觉得有问题的旧路径
# FILE_PATH = "/data/user/rencw/ICL-I2PReg/glb_objs/tube.blend" 
FILE_PATH = "/data/user/rencw/ICL-I2PReg/glb_objs/tube.blend" # <--- 建议先测这个

print(f"--- 正在尸检文件: {FILE_PATH} ---")

if not os.path.exists(FILE_PATH):
    print("错误: 文件根本不存在！请检查路径拼写。")
    sys.exit(1)

with bpy.data.libraries.load(FILE_PATH, link=False) as (data_from, data_to):
    print(f"文件内发现 {len(data_from.objects)} 个物体:")
    data_to.objects = data_from.objects

for obj in data_to.objects:
    if obj is None: continue
    print(f"\n[物体] 名称: {obj.name}")
    print(f"       类型: {obj.type}")
    
    if obj.type == 'MESH':
        # 这里读取的是纯硬盘数据，没有任何修改器干扰
        v_count = len(obj.data.vertices)
        print(f"       顶点数: {v_count}")
        if v_count == 0:
            print("       ⚠️ 这是一个空网格！")
        else:
            print("       ✅ 这是一个有效网格。")
    else:
        print("       (非网格物体)")

print("\n-----------------------------")