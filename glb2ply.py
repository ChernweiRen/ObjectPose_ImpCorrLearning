import bpy
import os
import sys
import glob

# ================= Configuration =================
# 输入 GLB 文件夹路径
# INPUT_DIR = "/data/user/rencw/ICL-I2PReg/glb_objs/test_tube_rack.glb"
INPUT_DIR = "/data/user/rencw/ICL-I2PReg/glb_objs/tube.glb"
# 输出 PLY 文件夹路径
OUTPUT_DIR = "/data/user/rencw/ICL-I2PReg/ply_objs/"
# =================================================

def convert_glb_to_ply_baked():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if INPUT_DIR.endswith("glb"):
        glb_files = [INPUT_DIR]
    else:
        glb_files = glob.glob(os.path.join(INPUT_DIR, "*.glb"))

    
    for glb_path in glb_files:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        filename = os.path.basename(glb_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        print(f"Processing: {filename}...")
        
        # 1. 导入并准备物体
        bpy.ops.import_scene.gltf(filepath=glb_path, merge_vertices=True)
        mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
        if not mesh_objs: continue
            
        bpy.context.view_layer.objects.active = mesh_objs[0]
        bpy.ops.object.select_all(action='DESELECT')
        for obj in mesh_objs: obj.select_set(True)
        if len(mesh_objs) > 1: bpy.ops.object.join()
        
        final_obj = bpy.context.view_layer.objects.active
        bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

        # ========================================================
        # 【新增步骤】将物体原点移动到几何中心 (Bounds Center)
        # ========================================================
        # 必须确保物体被选中且处于激活状态
        final_obj.select_set(True)
        bpy.context.view_layer.objects.active = final_obj
        # 执行归中操作
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        
        # 强制归零 (Raw State)
        # 注意：先改原点，再归零位置，这样物体就会处于世界坐标的 (0,0,0) 正中心
        final_obj.rotation_euler = (0, 0, 0)
        final_obj.location = (0, 0, 0)
        # final_obj.scale = (1, 1, 1) # 保持巨大尺寸
        print(final_obj.scale)
        # final_obj.scale = (final_obj.scale.x / 0.0254, final_obj.scale.y / 0.0254, final_obj.scale.z / 0.0254)
        # print(final_obj.scale)
        final_obj.scale = (final_obj.scale.x / 0.001, final_obj.scale.y / 0.001, final_obj.scale.z / 0.001) # set to mm
        print(final_obj.scale)
        # bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        base_path = os.path.join(OUTPUT_DIR, name_no_ext)

        # ========================================================
        # 导出 4 个变体 (穷举所有可能的轴向定义)
        # ========================================================
        
        # Variant A: Blender 原生 (Forward=Y, Up=Z)
        bpy.ops.export_mesh.ply(
            filepath=f"{base_path}_vA_Blender.ply",
            use_selection=True, global_scale=1.0,
            axis_forward='Y', axis_up='Z'
        )
        
        # Variant B: GLTF 标准 (Forward=Z, Up=Y)
        bpy.ops.export_mesh.ply(
            filepath=f"{base_path}_vB_GLTF.ply",
            use_selection=True, global_scale=1.0,
            axis_forward='Z', axis_up='Y'
        )
        
        # Variant C: 负 Z 前向 (Forward=-Z, Up=Y)
        bpy.ops.export_mesh.ply(
            filepath=f"{base_path}_vC_NegZ.ply",
            use_selection=True, global_scale=1.0,
            axis_forward='-Z', axis_up='Y'
        )
        
        # Variant D: 交换 YZ (Forward=Y, Up=-Z)
        bpy.ops.export_mesh.ply(
            filepath=f"{base_path}_vD_Swap.ply",
            use_selection=True, global_scale=1.0,
            axis_forward='Y', axis_up='-Z'
        )
        
        print(f"  -> Exported 4 variants for {filename}")

if __name__ == "__main__":
    convert_glb_to_ply_baked()