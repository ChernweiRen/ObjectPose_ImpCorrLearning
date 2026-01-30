import bpy
import os
import sys
import glob
import math
from mathutils import Matrix  # 【重要】需要引入 Matrix

# ================= Configuration =================
# 输入 .blend 文件路径
# INPUT_PATH = "/data/user/rencw/ICL-I2PReg/glb_objs/tube.blend"
INPUT_PATH = "/data/user/rencw/ICL-I2PReg/glb_objs/tube_thick.blend"
# 输出 PLY 文件夹路径
OUTPUT_DIR = "/data/user/rencw/ICL-I2PReg/ply_objs/"
# =================================================

# 【新增】与渲染脚本一致的 Root 查找逻辑
def find_root_object(objs):
    for o in objs:
        if o.parent is None: return o
    return objs[0]

# 【新增】递归查找子物体
def get_all_children_recursive(obj):
    children = []
    for c in obj.children:
        children.append(c)
        children.extend(get_all_children_recursive(c))
    return children

def convert_blend_to_ply_baked():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 确定要处理的文件列表
    if INPUT_PATH.endswith(".blend"):
        blend_files = [INPUT_PATH]
    else:
        blend_files = glob.glob(os.path.join(INPUT_PATH, "*.blend"))

    if not blend_files:
        print(f"No .blend files found in {INPUT_PATH}")
        return

    for blend_path in blend_files:
        # 重置 Blender 场景，保持干净
        bpy.ops.wm.read_factory_settings(use_empty=True)
        
        filename = os.path.basename(blend_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        print(f"Processing: {filename}...")
        
        # ========================================================
        # 导入 .blend 文件中的所有 Object
        # ========================================================
        try:
            with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
                data_to.objects = data_from.objects
        except Exception as e:
            print(f"Error loading {blend_path}: {e}")
            continue

        imported_objs = []
        for obj in data_to.objects:
            if obj is not None:
                bpy.context.collection.objects.link(obj)
                imported_objs.append(obj)
        
        # 【步骤 A】找到 Root Object (渲染脚本的基准)
        if not imported_objs: continue
        root_obj = find_root_object(imported_objs)
        print(f"  Root object determined: {root_obj.name}")
        
        # 获取 Root 的逆矩阵 (用于将世界坐标还原回 Root 局部坐标)
        root_inv_matrix = root_obj.matrix_world.inverted()

        # 【步骤 B】收集所有 Mesh
        all_family = [root_obj] + get_all_children_recursive(root_obj)
        mesh_objs = [o for o in all_family if o.type == 'MESH']
        
        if not mesh_objs: 
            print(f"Skipping {filename}: No mesh objects found.")
            continue
            
        # ========================================================
        # 几何处理 (合并、居中、缩放)
        # ========================================================
        
        # 【步骤 C】准备 Mesh：全部转为世界坐标下的绝对几何体
        bpy.ops.object.select_all(action='DESELECT')
        for obj in mesh_objs: 
            obj.select_set(True)
            # 必须先解除父子关系，并应用变换，确保 join 之前它们在视觉位置上是对的
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        
        # 【步骤 D】合并
        bpy.ops.object.select_all(action='DESELECT')
        for obj in mesh_objs: obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_objs[0]
        
        if len(mesh_objs) > 1: 
            bpy.ops.object.join()
        
        final_obj = bpy.context.view_layer.objects.active
        
        # 【步骤 E：核心修复】应用逆矩阵变换
        # 将网格数据从 "世界空间" 变换回 "Root 局部空间"
        # 这样，PLY 的 (0,0,0) 就绝对等同于 Root 的 Origin
        final_obj.data.transform(root_inv_matrix)
        
        # 变换后，必须强制归零 Object 的 Transform
        # 因为几何体顶点已经移动到了正确位置，容器必须归位
        final_obj.location = (0, 0, 0)
        final_obj.rotation_euler = (0, 0, 0)
        final_obj.scale = (1, 1, 1)
        
        # ----------------- 保留的注释代码 -----------------
        # bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

        # # set ORIGIN to the center of the bounding box
        # final_obj.select_set(True)
        # bpy.context.view_layer.objects.active = final_obj
        # bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        
        # final_obj.rotation_euler = (0, 0, 0)
        # final_obj.location = (0, 0, 0)
        # --------------------------------------------------
        
        # 【步骤 F】单位转换 (米 -> 毫米)
        # 此时 final_obj 是 Root 原始比例 (1:1)。
        # 你的验证脚本里写着 scale = 0.001，意味着 PLY 应该是毫米单位。
        print(f"Original Scale: {final_obj.scale}")
        
        # 直接设置为 1000，然后 Apply
        final_obj.scale = (1000.0, 1000.0, 1000.0) 
        print(f"New Scale: {final_obj.scale}")
        
        # 应用缩放，确保导出时顶点坐标真的是毫米级
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        base_path = os.path.join(OUTPUT_DIR, name_no_ext)

        # ========================================================
        # 导出 4 个变体 (适配 Blender 4.2+ 新 API)
        # ========================================================
        # 新 API: bpy.ops.wm.ply_export
        # 参数变更:
        #   use_selection -> export_selected_objects
        #   axis_forward -> forward_axis (必须是 'X', 'Y', 'Z', 'NEGATIVE_X', 'NEGATIVE_Y', 'NEGATIVE_Z')
        #   axis_up -> up_axis
        # ========================================================
        
        # Variant A: Blender 原生 (Forward=Y, Up=Z)
        # 既然我们已经还原到了 Root 的局部空间，直接用 Blender 原生轴向即可
        bpy.ops.wm.ply_export(
            filepath=f"{base_path}_vA_Blender.ply",
            export_selected_objects=True,
            global_scale=1.0,
            forward_axis='Y', up_axis='Z'
        )
        
        # Variant B: GLTF 标准 (Forward=Z, Up=Y)
        bpy.ops.wm.ply_export(
            filepath=f"{base_path}_vB_GLTF.ply",
            export_selected_objects=True,
            global_scale=1.0,
            forward_axis='Z', up_axis='Y'
        )
        
        # Variant C: 负 Z 前向 (Forward=-Z, Up=Y)
        bpy.ops.wm.ply_export(
            filepath=f"{base_path}_vC_NegZ.ply",
            export_selected_objects=True,
            global_scale=1.0,
            forward_axis='NEGATIVE_Z', up_axis='Y'
        )
        
        # Variant D: 交换 YZ (Forward=Y, Up=-Z)
        bpy.ops.wm.ply_export(
            filepath=f"{base_path}_vD_Swap.ply",
            export_selected_objects=True,
            global_scale=1.0,
            forward_axis='Y', up_axis='NEGATIVE_Z'
        )
        
        print(f"  -> Exported 4 variants for {filename}")

if __name__ == "__main__":
    convert_blend_to_ply_baked()