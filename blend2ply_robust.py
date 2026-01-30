import bpy
import os
import glob
from mathutils import Matrix
import bmesh

# ================= Configuration =================
INPUT_PATH = "/data/user/rencw/ICL-I2PReg/glb_objs/tube_thick.blend"
OUTPUT_DIR = "/data/user/rencw/ICL-I2PReg/ply_objs/"
# =================================================

# ========================================================
# 强制三角化 + 去除 UV / attributes + 应用变换
# ========================================================
def prepare_mesh_for_ply(obj):
    # 1. 确保选中
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # 2. 应用所有 modifier（尤其是 subsurf / boolean）
    for mod in obj.modifiers:
        try:
            bpy.ops.object.modifier_apply(modifier=mod.name)
        except Exception:
            pass

    # 3. 应用世界变换（位置、旋转、缩放）
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # 4. 清理 UV / 顶点属性
    mesh = obj.data
    while mesh.uv_layers:
        mesh.uv_layers.remove(mesh.uv_layers[0])
    for attr in list(mesh.attributes):
        if attr.is_required:
            continue
        mesh.attributes.remove(attr)

    # 5. BMesh triangulate
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    bmesh.ops.triangulate(
        bm,
        faces=bm.faces[:],
        quad_method='BEAUTY',
        ngon_method='BEAUTY'
    )
    bm.to_mesh(mesh)
    bm.free()

def find_root_object(objs):
    for o in objs:
        if o.parent is None:
            return o
    return objs[0]

def get_all_children_recursive(obj):
    children = []
    for c in obj.children:
        children.append(c)
        children.extend(get_all_children_recursive(c))
    return children

def convert_blend_to_ply_baked():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    blend_files = [INPUT_PATH] if INPUT_PATH.endswith(".blend") else glob.glob(os.path.join(INPUT_PATH, "*.blend"))

    if not blend_files:
        print(f"No .blend files found in {INPUT_PATH}")
        return

    for blend_path in blend_files:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        filename = os.path.basename(blend_path)
        name_no_ext = os.path.splitext(filename)[0]
        print(f"Processing: {filename}...")

        # 导入对象
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

        if not imported_objs: 
            continue

        root_obj = find_root_object(imported_objs)
        print(f"  Root object: {root_obj.name}")
        root_inv_matrix = root_obj.matrix_world.inverted()

        # 收集所有 mesh
        all_family = [root_obj] + get_all_children_recursive(root_obj)
        mesh_objs = [o for o in all_family if o.type == 'MESH']
        if not mesh_objs:
            print(f"Skipping {filename}: No mesh objects found.")
            continue

        # 应用变换到世界坐标
        bpy.ops.object.select_all(action='DESELECT')
        for obj in mesh_objs:
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # 合并 mesh
        bpy.ops.object.select_all(action='DESELECT')
        for obj in mesh_objs:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_objs[0]
        if len(mesh_objs) > 1:
            bpy.ops.object.join()
        final_obj = bpy.context.view_layer.objects.active

        # 逆矩阵变换回 Root 局部空间
        final_obj.data.transform(root_inv_matrix)
        final_obj.location = (0,0,0)
        final_obj.rotation_euler = (0,0,0)
        final_obj.scale = (1,1,1)

        # 缩放到毫米
        final_obj.scale = (1000, 1000, 1000)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        final_obj = bpy.context.view_layer.objects.active
        prepare_mesh_for_ply(final_obj)

        # 导出 PLY (Blender 4.4 不传 export_uv/export_normals/export_colors)
        base_path = os.path.join(OUTPUT_DIR, name_no_ext)
        for suffix, fwd, up in [
            ("vA_Blender", "Y", "Z"),
            ("vB_GLTF", "Z", "Y"),
            ("vC_NegZ", "NEGATIVE_Z", "Y"),
            ("vD_Swap", "Y", "NEGATIVE_Z"),
        ]:
            bpy.ops.wm.ply_export(
                filepath=f"{base_path}_{suffix}.ply",
                export_selected_objects=True,
                global_scale=1.0,
                forward_axis=fwd,
                up_axis=up
            )
        print(f"  -> Exported 4 variants for {filename}")

if __name__ == "__main__":
    convert_blend_to_ply_baked()