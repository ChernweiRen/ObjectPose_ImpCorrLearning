"""Blender script to render images of 3D models.
Fixes:
1. Supports .blend file loading explicitly.
2. Fixes 'Fixed object not found' error by loose name matching.
3. Fixes Segmentation Fault by disabling Denoising and activating GPU explicitly.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from tqdm import tqdm

from mathutils import Vector, Matrix
import numpy as np

import bpy
from contextlib import contextmanager

# ==========================================
# Configs & Utils
# ==========================================
@contextmanager
def suppress_output():
    try:
        null_fd = os.open(os.devnull, os.O_RDWR)
        save_stdout = os.dup(1)
        save_stderr = os.dup(2)
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        yield
    except Exception:
        yield
    finally:
        try:
            os.dup2(save_stdout, 1)
            os.dup2(save_stderr, 2)
            os.close(null_fd)
        except Exception:
            pass
        
# defined random process functions
def randomize_glass_ior(objects, mat_target_name="Glass.002", min_ior=1.0, max_ior=1.5):
    """
    遍历物体材质，找到 Glass BSDF 并随机化 IOR。
    """
    
    modified_count = 0
    new_ior = random.uniform(min_ior, max_ior)
    
    for obj in objects:
        # if obj.type != 'MESH' or target_name != obj.name: continue
        if obj.type != 'MESH': continue

        
        # 遍历该物体所有的材质槽
        for slot in obj.material_slots:
            if mat_target_name != slot.material.name: continue
            mat = slot.material
            if mat and mat.use_nodes:
                # 遍历节点树寻找 Glass BSDF
                for node in mat.node_tree.nodes:
                    # 目标明确：Glass BSDF
                    if node.type == 'BSDF_GLASS':
                        # 修改 IOR
                        node.inputs['IOR'].default_value = new_ior
                        # 修改粗糙度 (可选，增加一点磨砂感的变化)
                        # node.inputs['Roughness'].default_value = random.uniform(0.0, 0.2)
                        modified_count += 1
                        
                    # 兼容：如果是 Principled BSDF (原理化) 且开启了透射
                    elif node.type == 'BSDF_PRINCIPLED':
                        # Blender 4.0+ Transmission 叫做 'Transmission Weight'
                        trans_input = node.inputs.get('Transmission Weight') or node.inputs.get('Transmission')
                        if trans_input and trans_input.default_value > 0:
                            node.inputs['IOR'].default_value = new_ior
                            modified_count += 1

    if modified_count > 0:
        print(f"  [Material] Randomized Glass IOR to {new_ior:.3f} for {modified_count} slots: {objects}")

def randomize_water_height(objects, target_name="tube.002", min_scale=0.0, max_scale=0.084):
    """
    找到指定名字的水体物体，并随机调整其 Y 轴缩放。
    """
    
    found = False
    new_scale = random.uniform(min_scale, max_scale)
    
    for obj in objects:
        # 使用 in 判断，防止 Blender 自动重命名 (例如 tube.002.001)
        if target_name in obj.name:
            # if reset_origin:
            #     bpy.context.scene.cursor.location = Vector(reset_origin)
            #     bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
            #     print(f"  [Water] obj.location: {obj.location}, obj.matrix_world.translation: {obj.matrix_world.translation}")
            # 修改 Y 轴缩放 (Index 1)
            # 注意：前提是物体的原点(Origin)在底部。如果原点在中心，水会两头缩。
            obj.scale[1] = new_scale
            print(f"  [Water] Set '{obj.name}' scale.y to {new_scale:.5f}")
            found = True
            
    if not found:
        print(f"  [Water] Warning: Could not find water object with name '{target_name}'")
        
        
        



TABLE_CONFIG = {
    "plane_checker.glb": [-8., -8., 8., 8.],
    "plane_furn_cabinet.glb": [-7.96, -3.46, 9.1, 3.46],
    "plane_furn.glb": [7.32],
    "plane_gray.glb": [-4.0, -4.0, 4.0, 4.0],
    "plane_wood.glb": [-2.96, -3.98, 2.96, 3.98],
    "plane_table.glb": [-4.6, -12.68, 4.6, 12.68],
}
TABLE_LIST = list(TABLE_CONFIG.keys())

# 【修改点 1】明确指定你要用的 .blend 文件名
# fixed_obj_names = "tube.blend"
# fixed_obj_names = "tube_water.blend"
fixed_obj_names_list = ["tube.blend", "tube_water.blend"]
fixed_obj_names = random.choice(fixed_obj_names_list)
fixed_obj_rand_proc_func = {
    "tube.blend": {randomize_glass_ior: ["Glass.002", 1.1, 2.4], },
    "tube_water.blend": {randomize_water_height: ["tube.002", 0.0, 0.084], randomize_glass_ior: ["Glass.002", 1.1, 2.4]},
}
    
OBJECT_LIST = [] # 其他干扰物体，如果需要可以加





def get_all_children(obj):
    children = []
    for c in obj.children:
        children.append(c)
        children.extend(get_all_children(c))
    return children

def get_world_bbox(obj):
    bbox_min = Vector((math.inf, math.inf, math.inf))
    bbox_max = Vector((-math.inf, -math.inf, -math.inf))
    meshes = []
    if obj.type == 'MESH': meshes.append(obj)
    for c in get_all_children(obj):
        if c.type == 'MESH': meshes.append(c)
    if obj.type == 'EMPTY' and obj.instance_type == 'COLLECTION':
        for c in obj.instance_collection.objects:
            if c.type == 'MESH': meshes.append(c)
            for cc in get_all_children(c):
                if cc.type == 'MESH': meshes.append(cc)
    if not meshes: return Vector((-0.1, -0.1, -0.1)), Vector((0.1, 0.1, 0.1))
    for m in meshes:
        for v in m.bound_box:
            p = m.matrix_world @ Vector(v)
            bbox_min = Vector((min(bbox_min[0], p[0]), min(bbox_min[1], p[1]), min(bbox_min[2], p[2])))
            bbox_max = Vector((max(bbox_max[0], p[0]), max(bbox_max[1], p[1]), max(bbox_max[2], p[2])))
    return bbox_min, bbox_max

def bbox_overlap_2d(bbox_a, bbox_b):
    (ax_min, ay_min), (ax_max, ay_max) = bbox_a
    (bx_min, by_min), (bx_max, by_max) = bbox_b
    return not (ax_max <= bx_min or ax_min >= bx_max or ay_max <= by_min or ay_min >= by_max)

def get_scene_objects_set(): return set(bpy.context.scene.objects)
def get_new_objects(old_set): return [o for o in bpy.context.scene.objects if o not in old_set]
def get_meshes_from_objects(objs):
    meshes = []
    for o in objs:
        if o.type == 'MESH': meshes.append(o)
        if o.type == 'EMPTY' and o.instance_type == 'COLLECTION':
            for c in o.instance_collection.objects:
                if c.type == 'MESH': meshes.append(c)
    return meshes

def get_world_bbox_from_meshes(meshes):
    if not meshes: raise RuntimeError("No mesh found")
    bbox_min = Vector((math.inf, math.inf, math.inf))
    bbox_max = Vector((-math.inf, -math.inf, -math.inf))
    for m in meshes:
        for v in m.bound_box:
            p = m.matrix_world @ Vector(v)
            bbox_min.x = min(bbox_min.x, p.x); bbox_min.y = min(bbox_min.y, p.y); bbox_min.z = min(bbox_min.z, p.z)
            bbox_max.x = max(bbox_max.x, p.x); bbox_max.y = max(bbox_max.y, p.y); bbox_max.z = max(bbox_max.z, p.z)
    return bbox_min, bbox_max

def find_root_object(objs):
    for o in objs:
        if o.parent is None: return o
    return objs[0]

def reset_scene() -> None:
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}: bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials: bpy.data.materials.remove(material, do_unlink=True)
    for texture in bpy.data.textures: bpy.data.textures.remove(texture, do_unlink=True)
    for image in bpy.data.images: bpy.data.images.remove(image, do_unlink=True)

# 【修改点 2】健壮的加载函数，支持 blend/glb
def load_object(object_path: str) -> None:
    print(f"\n[DEBUG] Opening File: {object_path}")
    if not os.path.exists(object_path):
        print(f"[ERROR] File not found: {object_path}")
        return

    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".blend"):
        with bpy.data.libraries.load(object_path, link=False) as (data_from, data_to):
            data_to.objects = data_from.objects

        linked_count = 0
        for obj in data_to.objects:
            if obj is not None:
                # 即使是 0 顶点也 Link 进来，交给主循环去评估修改器
                if obj.name not in bpy.context.scene.objects:
                    try:
                        bpy.context.collection.objects.link(obj)
                        linked_count += 1
                    except RuntimeError:
                        pass
        print(f"[DEBUG] Linked {linked_count} objects from blend file.")
    else: 
        raise ValueError(f"Unsupported file type: {object_path}")

# ==========================================
# Material & Compositor Setup
# ==========================================
def create_emission_material(name, color=(1, 1, 1, 1)):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes; links = mat.node_tree.links; nodes.clear()
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs['Color'].default_value = color
    out = nodes.new(type="ShaderNodeOutputMaterial")
    links.new(emission.outputs[0], out.inputs[0])
    return mat

def create_opaque_material(name, color=(0.5, 0.5, 0.5, 1)):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes; links = mat.node_tree.links; nodes.clear()
    bsdf = nodes.new(type="ShaderNodeBsdfDiffuse")
    bsdf.inputs['Color'].default_value = color
    out = nodes.new(type="ShaderNodeOutputMaterial")
    links.new(bsdf.outputs[0], out.inputs[0])
    return mat

def setup_compositor_nodes(scene):
    scene.render.use_compositing = True
    scene.use_nodes = True
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links
    nodes.clear()
    
    scene.view_layers[0].use_pass_z = True

    rl_node = nodes.new(type="CompositorNodeRLayers")
    
    depth_out = nodes.new(type="CompositorNodeOutputFile")
    depth_out.name = "Depth_Output"
    depth_out.format.file_format = 'OPEN_EXR'
    depth_out.format.color_depth = '32'
    depth_out.file_slots[0].path = "depth" 
    
    mask_out = nodes.new(type="CompositorNodeOutputFile")
    mask_out.name = "Mask_Output"
    mask_out.format.file_format = 'PNG'
    mask_out.format.color_mode = 'BW'
    mask_out.file_slots[0].path = "mask"

    links.new(rl_node.outputs['Depth'], depth_out.inputs[0])
    links.new(rl_node.outputs['Alpha'], mask_out.inputs[0])
    
    depth_out.mute = True
    mask_out.mute = True
    return depth_out, mask_out






# ==========================================
# Main Script
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--object_path", type=str, required=True)
parser.add_argument("--plane_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--engine", type=str, default="CYCLES")
parser.add_argument("--camera_dist", type=float, default=1.2)
parser.add_argument("--camera_dist_min", type=float, default=1.2)
parser.add_argument("--camera_dist_max", type=float, default=2.2)
parser.add_argument("--elevation", type=float, default=30)
parser.add_argument("--elevation_min", type=float, default=-10)
parser.add_argument("--elevation_max", type=float, default=40)
parser.add_argument("--num_images", type=int, default=16)
parser.add_argument("--res_w", type=int, default=256)
parser.add_argument("--res_h", type=int, default=256)
parser.add_argument("--hdrs_dir", type=str, default="/data/user/rencw/ICL-I2PReg/hdri_bgs")
parser.add_argument("--auto_offset", type=bool, default=False)
parser.add_argument("--normalize_scene", type=bool, default=False)
parser.add_argument("--device", type=str, default='CUDA')
parser.add_argument("--camera_type", type=str, default="random", choices=["random", "fixed"])

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

context = bpy.context
scene = context.scene
render = scene.render

# Camera setup
cam = scene.objects["Camera"]
cam.location = (0, 0, 0); cam.data.lens = 35; cam.data.sensor_width = 32
cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"; cam_constraint.up_axis = "UP_Y"

# ==========================================
# 【修改点 3】 渲染与 GPU 设置 (防崩关键)
# ==========================================
render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = args.res_w; render.resolution_y = args.res_h

scene.cycles.device = "GPU"
# 1. 强制关闭降噪 (Headless 模式下 OIDN 极易导致 Segfault)
# scene.cycles.use_denoising = False 
scene.cycles.use_denoising = True
scene.cycles.denoiser = "OPTIX"
scene.cycles.samples = 512

# 2. 显式激活 CUDA 设备
cprefs = bpy.context.preferences.addons['cycles'].preferences
cprefs.compute_device_type = args.device # 'CUDA'
cprefs.get_devices()
print(f"[GPU Setup] Activating {args.device} devices:")
for device in cprefs.devices:
    if device.type == args.device:
        device.use = True
        print(f"  - Activated: {device.name}")

# Math Utils
def az_el_to_points(azimuths, elevations):
    x = np.cos(azimuths)*np.cos(elevations)
    y = np.sin(azimuths)*np.cos(elevations)
    z = np.sin(elevations)
    return np.stack([x,y,z],-1)

def set_camera_location(cam_pt):
    x, y, z = cam_pt
    bpy.data.objects["Camera"].location = x, y, z
    return bpy.data.objects["Camera"]

def get_3x4_RT_matrix_from_blender(cam):
    bpy.context.view_layer.update()
    location, rotation = cam.matrix_world.decompose()[0:2]
    R = np.asarray(rotation.to_matrix()); t = np.asarray(location)
    cam_rec = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float64)
    R = R.T; t = -R @ t
    return np.concatenate([cam_rec @ R, (cam_rec @ t)[:,None]], 1)

def get_calibration_matrix_K_from_blender(camera):
    f_mm = camera.data.lens
    scale = scene.render.resolution_percentage / 100
    w_mm = camera.data.sensor_width; h_mm = camera.data.sensor_height
    res_x = scene.render.resolution_x; res_y = scene.render.resolution_y
    pixel_aspect = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if camera.data.sensor_fit == 'VERTICAL':
        s_u = res_x * scale / w_mm / pixel_aspect; s_v = res_y * scale / h_mm
    else:
        s_u = res_x * scale / w_mm; s_v = res_y * scale * pixel_aspect / h_mm
    return np.asarray(((f_mm*s_u, 0, res_x*scale/2), (0, f_mm*s_u, res_y*scale/2), (0, 0, 1)), np.float64)

# ==========================================
# Main
# ==========================================
def save_images():
    out_dir_path = Path(args.output_dir).resolve() 
    os.makedirs(out_dir_path, exist_ok=True)
    
    reset_scene()
    
    mat_opaque = create_opaque_material("Mat_Opaque", color=(0.5, 0.5, 0.5, 1))
    mat_white = create_emission_material("Mat_White", color=(1, 1, 1, 1))
    
    depth_out_node, mask_out_node = setup_compositor_nodes(scene)
    depth_out_node.base_path = str(out_dir_path)
    mask_out_node.base_path = str(out_dir_path)

    # 1. Compose Scene
    table_name = random.choice(TABLE_LIST)
    table_glb = os.path.join(args.plane_path, table_name)

    num_objs = random.randint(0, 2)
    num_objs = min(num_objs, len(OBJECT_LIST))
    obj_names = random.sample(OBJECT_LIST, num_objs)
    # Put target first
    obj_names = [fixed_obj_names] + obj_names
    obj_glbs = [os.path.join(args.object_path, n) for n in obj_names]

    print(f"[Compose] Table: {table_name} | Objects: {obj_names}")

    # Load Table
    objs_before_table = get_scene_objects_set()
    load_object(table_glb)
    bpy.context.view_layer.update()
    table_objs = get_new_objects(objs_before_table)
    table_bbox_cfg = TABLE_CONFIG.get(table_name, [-5, -5, 5, 5])
    
    placed_bboxes = []
    fixed_obj_parts = []
    other_obj_parts = []
    fixed_obj_root_ref = None

    # Load & Place Objects
    for obj_glb in obj_glbs:
        old_objs = get_scene_objects_set()
        
        try:
            load_object(obj_glb)
        except Exception as e:
            print(f"[Error] Load failed: {e}")
            continue

        bpy.context.view_layer.update()
        current_new_objs = get_new_objects(old_objs)

        if not current_new_objs:
            # Check if this missing object was our target
            if fixed_obj_names in obj_glb:
                print(f"[Critical] Target object '{fixed_obj_names}' yielded no objects! Aborting.")
                return
            continue

        # Evaluate modifiers (for Geometry Nodes or tricky meshes)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        valid_mesh_found = False

        # bpy.ops.object.select_all(action='DESELECT')
        # for obj in current_new_objs:
        #     if obj.type == 'MESH':
        #         obj_eval = obj.evaluated_get(depsgraph)
        #         if len(obj_eval.data.vertices) > 0:
        #             valid_mesh_found = True
        #             # Safe Origin Set
        #             obj.select_set(True)
        #             bpy.context.view_layer.objects.active = obj 
        #             try:
        #                 bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        #             except: pass
        #             obj.select_set(False)

        bpy.context.view_layer.update()

        root = find_root_object(current_new_objs)
        meshes = get_meshes_from_objects(current_new_objs)
        
        # 【修改点 4】宽松的匹配逻辑 (只要包含名字就算，不论路径)
        if fixed_obj_names in obj_glb:
            print(f"  [Info] Identified target: {obj_glb}")

            # === 新增：随机调整 IOR ===
            # 范围建议：
            # 1.05 (空气/极薄) ~ 1.33 (水) ~ 1.5 (玻璃) ~ 1.8 (重火石玻璃) ~ 2.4 (钻石)
            # 建议 1.2 到 1.8 之间效果最自然
            # randomize_glass_ior(current_new_objs, min_ior=1.1, max_ior=2.4)
            # ========================
            
            #【新增】随机水位高度
            # 这里的 "tube.002" 必须和你 Blender 里看到的名字一致
            # randomize_water_height(current_new_objs, target_name="tube.002", min_scale=0.0, max_scale=0.084)
            
            # random process functions
            for func, params in fixed_obj_rand_proc_func[fixed_obj_names].items():
                func(current_new_objs, *params)
            
            fixed_obj_parts.extend(current_new_objs)
            fixed_obj_root_ref = root
        else:
            other_obj_parts.extend(current_new_objs)

        placed = False
        for _ in range(100):
            if len(table_bbox_cfg) == 4:
                xmin, ymin, xmax, ymax = table_bbox_cfg
                x = random.uniform(xmin, xmax); y = random.uniform(ymin, ymax)
            else:
                R = table_bbox_cfg[0]
                r = random.uniform(0, R); theta = random.uniform(0, 2 * math.pi)
                x = r * math.cos(theta); y = r * math.sin(theta)

            root.location = (x, y, 0.0)
            bpy.context.view_layer.update()
            
            if meshes:
                bbox_min, bbox_max = get_world_bbox_from_meshes(meshes)
                root.location.z = -bbox_min.z
                bpy.context.view_layer.update()
                bbox_min, bbox_max = get_world_bbox_from_meshes(meshes)
                bbox2d = ((bbox_min.x, bbox_min.y), (bbox_max.x, bbox_max.y))

                if not any(bbox_overlap_2d(bbox2d, b) for b in placed_bboxes):
                    placed_bboxes.append(bbox2d)
                    placed = True
                    print(f"Placed {os.path.basename(obj_glb)} successfully.")
                    break
            else:
                placed = True
                break
    
        if not placed: 
            print(f"[Warning] Failed to place {obj_glb}")
            for o in current_new_objs: bpy.data.objects.remove(o, do_unlink=True)
            if fixed_obj_names in obj_glb: return

    # Check if target was found
    if fixed_obj_root_ref is None:
        print(f"[Critical] Fixed object '{fixed_obj_names}' was not found in loaded objects! Check filenames.")
        return

    world2obj_matrix = np.array(fixed_obj_root_ref.matrix_world.inverted())

    # Lighting
    world = bpy.context.scene.world; world.use_nodes = True
    nodes = world.node_tree.nodes; nodes.clear()
    env_tex = nodes.new(type="ShaderNodeTexEnvironment")
    env_tex.image = bpy.data.images.load(os.path.join(args.hdrs_dir, random.choice(os.listdir(args.hdrs_dir))))
    bg = nodes.new(type="ShaderNodeBackground"); bg.inputs["Strength"].default_value = 1.0
    out = nodes.new(type="ShaderNodeOutputWorld")
    world.node_tree.links.new(env_tex.outputs["Color"], bg.inputs["Color"])
    world.node_tree.links.new(bg.outputs["Background"], out.inputs["Surface"])

    # Cam Constraint
    empty = bpy.data.objects.new("Empty", None); bpy.context.scene.collection.objects.link(empty)
    cam_constraint.target = empty

    # Output Dir
    scene_id = f"{table_name}__{'_'.join(obj_names)}"
    out_dir_scene = out_dir_path / scene_id
    if not os.path.exists(out_dir_scene):
        out_dir_scene.mkdir(parents=True, exist_ok=True)
    else:
        for i in range(100):
            new_out_dir_scene = Path(str(out_dir_scene) + f"_{i}")
            if not os.path.exists(new_out_dir_scene):
                new_out_dir_scene.mkdir(parents=True, exist_ok=True)
                out_dir_scene = new_out_dir_scene
                break
    print(f"Output directory: {out_dir_scene}")
    
    depth_out_node.base_path = str(out_dir_scene)
    mask_out_node.base_path = str(out_dir_scene)

    # Camera Poses
    if args.camera_type == "random":
        distances = np.random.uniform(args.camera_dist_min, args.camera_dist_max, args.num_images)
        azimuths = np.random.uniform(0, 2*np.pi, args.num_images)
        elevations = np.deg2rad(np.random.uniform(args.elevation_min, args.elevation_max, args.num_images))
    elif args.camera_type == "fixed":
        distances = np.full(args.num_images, args.camera_dist); azimuths = np.linspace(0, 2*np.pi, args.num_images, endpoint=False)
        elevations = np.deg2rad(np.full(args.num_images, args.elevation))
    cam_pts = az_el_to_points(azimuths, elevations) * distances[:, None]
    
    cam_poses = []
    obj2cam_poses = []
    
    def rename_output(folder, prefix, index, ext=".png"):
        src = folder / f"{prefix}0001{ext}"
        dst = folder / f"{prefix}{ext}"
        for _ in range(10): 
            if src.exists():
                if dst.exists(): os.remove(dst)
                os.rename(src, dst)
                return
            time.sleep(0.05)

    for i in tqdm(range(args.num_images)):
        scene.frame_set(1)
        camera = set_camera_location(cam_pts[i])
        RT = get_3x4_RT_matrix_from_blender(camera); cam_poses.append(RT)
        RT_4x4 = np.concatenate([RT, np.zeros((1, 4))], 0); RT_4x4[-1, -1] = 1
        obj2cam_poses.append(RT_4x4 @ np.linalg.inv(world2obj_matrix))

        # PASS 1: RGB
        scene.view_layers[0].material_override = None 
        bpy.context.view_layer.update()
        for o in table_objs: o.hide_render = False
        for o in fixed_obj_parts: o.hide_render = False
        for o in other_obj_parts: o.hide_render = False
        scene.render.film_transparent = False 
        depth_out_node.mute = True; mask_out_node.mute = True 
        scene.render.filepath = str(out_dir_scene / f"{i:03d}.png")
        
        # 【修改点 5】 渲染 RGB (这里最容易崩)
        bpy.ops.render.render(write_still=True)

        # PASS 2: Depth
        scene.view_layers[0].material_override = mat_opaque
        bpy.context.view_layer.update()
        scene.render.film_transparent = True 
        depth_out_node.mute = False; mask_out_node.mute = True 
        f_depth = f"{i:03d}_depth"
        depth_out_node.file_slots[0].path = f_depth
        scene.render.filepath = str(out_dir_scene / "temp_pass.png")
        bpy.ops.render.render(write_still=True)
        rename_output(out_dir_scene, f_depth, i, ".exr")

        # Clean override
        scene.view_layers[0].material_override = None
        bpy.context.view_layer.update()

        # PASS 3: Mask
        for o in table_objs: o.hide_render = True
        for o in other_obj_parts: o.hide_render = True
        for o in fixed_obj_parts: o.hide_render = False
        scene.view_layers[0].material_override = mat_white
        bpy.context.view_layer.update()
        depth_out_node.mute = True; mask_out_node.mute = False
        f_mask = f"{i:03d}_mask"
        mask_out_node.file_slots[0].path = f_mask
        bpy.ops.render.render(write_still=True)
        rename_output(out_dir_scene, f_mask, i, ".png")
        
        # Clean override
        scene.view_layers[0].material_override = None
        bpy.context.view_layer.update()
        
        if os.path.exists(str(out_dir_scene / "temp_pass.png")):
            os.remove(str(out_dir_scene / "temp_pass.png"))

    K = get_calibration_matrix_K_from_blender(camera)
    for i in range(args.num_images):
        np.savez(out_dir_scene / f"{i:03d}.npz", table=table_name, objects=obj_names, K=K, RT=cam_poses[i], obj2cam_poses=obj2cam_poses[i], azimuth=azimuths[i], elevation=elevations[i], distance=distances[i])

if __name__ == "__main__":
    save_images()