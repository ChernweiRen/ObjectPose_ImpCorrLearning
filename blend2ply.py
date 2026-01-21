import bpy
import os
import sys
import glob

# ================= Configuration =================
# 输入 .blend 文件路径
INPUT_PATH = "/data/user/rencw/ICL-I2PReg/glb_objs/tube.blend" 
# 输出 PLY 文件夹路径
OUTPUT_DIR = "/data/user/rencw/ICL-I2PReg/ply_objs/"
# =================================================

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
        
        mesh_objs = [o for o in imported_objs if o.type == 'MESH']
        
        if not mesh_objs: 
            print(f"Skipping {filename}: No mesh objects found.")
            continue
            
        # ========================================================
        # 几何处理 (合并、居中、缩放)
        # ========================================================
        
        bpy.context.view_layer.objects.active = mesh_objs[0]
        bpy.ops.object.select_all(action='DESELECT')
        for obj in mesh_objs: 
            obj.select_set(True)
        
        if len(mesh_objs) > 1: 
            bpy.ops.object.join()
        
        final_obj = bpy.context.view_layer.objects.active
        bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

        final_obj.select_set(True)
        bpy.context.view_layer.objects.active = final_obj
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        
        final_obj.rotation_euler = (0, 0, 0)
        final_obj.location = (0, 0, 0)
        
        print(f"Original Scale: {final_obj.scale}")
        final_obj.scale = (final_obj.scale.x / 0.001, final_obj.scale.y / 0.001, final_obj.scale.z / 0.001) # set to mm
        print(f"New Scale: {final_obj.scale}")
        
        # 应用缩放，确保导出时顶点坐标真的是毫米级
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

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