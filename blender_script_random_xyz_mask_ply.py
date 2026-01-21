"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

Here, input_model_paths.json is a json file containing a list of paths to .glb.
"""

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
from pathlib import Path
from tqdm import tqdm

from mathutils import Vector, Matrix
import numpy as np

import bpy
from mathutils import Vector
import pickle
from contextlib import contextmanager

# ==========================================
# Context Manager for suppressing output
# ==========================================
@contextmanager
def suppress_output():
    try:
        # Linux/Mac/Unix
        null_fd = os.open(os.devnull, os.O_RDWR)
        save_stdout = os.dup(1)
        save_stderr = os.dup(2)
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        yield
    except Exception:
        # Fallback if specific OS calls fail (e.g. strictly Windows permission issues)
        yield
    finally:
        try:
            os.dup2(save_stdout, 1)
            os.dup2(save_stderr, 2)
            os.close(null_fd)
        except Exception:
            pass

# ==========================================
# Utility Functions
# ==========================================
def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, pkl_path):
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

# ==========================================
# Configs
# ==========================================
TABLE_CONFIG = {
    "plane_checker": [-8., -8., 8., 8.],
    "plane_furn_cabinet": [-7.96, -3.46, 9.1, 3.46],
    "plane_furn": [7.32],
    "plane_gray": [-4.0, -4.0, 4.0, 4.0],
    "plane_wood": [-2.96, -3.98, 2.96, 3.98],
    "plane_table": [-4.6, -12.68, 4.6, 12.68],
}

TABLE_LIST = list(TABLE_CONFIG.keys())

fixed_obj_names = "test_tube_rack" # The target object for XYZ/Mask

OBJECT_LIST = [
    "dropper_rack",
    "alarm_clock",
    "coffee_cup",
    "mass_chalice",
]

# ==========================================
# Helper Functions
# ==========================================
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

    if not meshes:
        # Fallback for empty objects without mesh children
        return Vector((-0.1, -0.1, -0.1)), Vector((0.1, 0.1, 0.1))

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

def get_scene_objects_set():
    return set(bpy.context.scene.objects)

def get_new_objects(old_set):
    return [o for o in bpy.context.scene.objects if o not in old_set]

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
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)
        
        
# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")



# ==========================================
# New Helper Functions for XYZ/Mask
# ==========================================

def create_coord_material():
    """Creates a material that outputs Object Coordinates as emission color."""
    mat = bpy.data.materials.new(name="ObjectCoordMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Create nodes
    # 1. Texture Coordinate: Gets the local object coordinates (XYZ)
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    
    # 2. Emission: Emits the coordinate values directly as color
    emission = nodes.new(type="ShaderNodeEmission")
    
    # 3. Output
    material_output = nodes.new(type="ShaderNodeOutputMaterial")

    # Link Object Coords -> Emission Color -> Surface
    links.new(tex_coord.outputs['Object'], emission.inputs['Color'])
    links.new(emission.outputs['Emission'], material_output.inputs['Surface'])
    
    return mat

def set_hierarchy_visibility(root_obj, visible=True):
    """Recursively set visibility for an object and its children."""
    root_obj.hide_render = not visible
    for child in root_obj.children:
        set_hierarchy_visibility(child, visible)

def setup_compositor_nodes(scene):
    """Setup compositor to save XYZ (EXR) and Mask (PNG) simultaneously."""
    scene.use_nodes = True
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links
    nodes.clear()

    # Input: Render Layers
    rl_node = nodes.new(type="CompositorNodeRLayers")
    
    # --- Node 1: XYZ Output (EXR) ---
    xyz_out = nodes.new(type="CompositorNodeOutputFile")
    xyz_out.name = "XYZ_Output"
    xyz_out.format.file_format = 'OPEN_EXR'
    xyz_out.format.color_depth = '32' # Float precision
    
    # 【修复重点】不要删除默认插槽，直接修改它
    # 默认插槽通常叫 "Image"，我们把它改成 "xyz"
    xyz_out.file_slots[0].path = "xyz"
    
    # --- Node 2: Mask Output (PNG) ---
    mask_out = nodes.new(type="CompositorNodeOutputFile")
    mask_out.name = "Mask_Output"
    mask_out.format.file_format = 'PNG'
    mask_out.format.color_mode = 'BW' # Binary mask
    
    # 【修复重点】直接修改默认插槽
    mask_out.file_slots[0].path = "mask"
    
    # Linking
    # Connect Image (XYZ Color) to XYZ Output
    links.new(rl_node.outputs['Image'], xyz_out.inputs[0])
    
    # Connect Alpha (Mask) to Mask Output
    links.new(rl_node.outputs['Alpha'], mask_out.inputs[0])
    
    # Mute them by default (so they don't run during RGB render)
    xyz_out.mute = True
    mask_out.mute = True
    
    return xyz_out, mask_out


# ==========================================
# Main Script
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--object_path", type=str, required=True)
parser.add_argument("--plane_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--engine", type=str, default="CYCLES")

# camera config
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

print(f'=================== {args.engine} ===================')

context = bpy.context
scene = context.scene
render = scene.render

# Camera Setup
cam = scene.objects["Camera"]
cam.location = (0, 0, 0)
cam.data.lens = 35
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"

# Render Settings
render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = args.res_w
render.resolution_y = args.res_h
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 512
scene.cycles.use_adaptive_sampling = True
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 1.0
scene.cycles.pixel_filter_type = 'GAUSSIAN'
scene.cycles.use_denoising = True
scene.cycles.denoiser = 'OPENIMAGEDENOISE'
scene.render.film_transparent = True # Important for Mask generation!

bpy.context.preferences.addons["cycles"].preferences.get_devices()
bpy.context.preferences.addons["cycles"].preferences.compute_device_type = args.device
bpy.context.scene.cycles.tile_size = 8192

# Math Utils
def az_el_to_points(azimuths, elevations):
    x = np.cos(azimuths)*np.cos(elevations)
    y = np.sin(azimuths)*np.cos(elevations)
    z = np.sin(elevations)
    return np.stack([x,y,z],-1)

def set_camera_location(cam_pt):
    x, y, z = cam_pt
    camera = bpy.data.objects["Camera"]
    camera.location = x, y, z
    return camera

def get_3x4_RT_matrix_from_blender(cam):
    bpy.context.view_layer.update()
    location, rotation = cam.matrix_world.decompose()[0:2]
    R = np.asarray(rotation.to_matrix())
    t = np.asarray(location)
    cam_rec = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float64)
    R = R.T
    t = -R @ t
    R_world2cv = cam_rec @ R
    t_world2cv = cam_rec @ t
    RT = np.concatenate([R_world2cv,t_world2cv[:,None]],1)
    return RT

def get_calibration_matrix_K_from_blender(camera):
    f_in_mm = camera.data.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camera.data.sensor_width
    sensor_height_in_mm = camera.data.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    if camera.data.sensor_fit == 'VERTICAL':
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_u
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0
    K = np.asarray(((alpha_u, skew, u_0), (0, alpha_v, v_0), (0, 0, 1)), np.float64)
    return K

# ==========================================
# Main Execution Function
# ==========================================
def save_images():

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 0. Setup Special Materials and Compositor
    # 注意：reset_scene 会删除材质，所以这里先 reset 再创建材质
    reset_scene() # <--- 先 Reset
    
    coord_mat = create_coord_material() # <--- 再创建
    xyz_out_node, mask_out_node = setup_compositor_nodes(scene)

    # 1. Randomly choose table & objects
    table_name = random.choice(TABLE_LIST)
    table_glb = os.path.join(args.plane_path, table_name + ".glb")

    num_objs = random.randint(0, 2)
    obj_names = random.sample(OBJECT_LIST, num_objs)
    obj_names = [fixed_obj_names] + obj_names
    obj_glbs = [os.path.join(args.object_path, n + ".glb") for n in obj_names]

    print(f"[Compose] Table: {table_name}")
    print(f"[Compose] Objects: {obj_names}")

    # ==========================================
    # 3. Load table (Improved Tracking)
    # ==========================================
    # 记录加载桌子前的所有物体
    objs_before_table = get_scene_objects_set()
    load_object(table_glb)
    bpy.context.view_layer.update()
    
    # 获取桌子导入的所有新物体（不仅仅是根节点）
    table_objs = get_new_objects(objs_before_table)
    
    # 为了布局计算，我们还是需要找到一个根节点来移动位置（通常桌子不需要动，但保持逻辑一致）
    # 如果桌子不需要随机移动，这一步其实可以跳过，但在你的逻辑里似乎要根据 bbox 放物体
    
    table_bbox_cfg = TABLE_CONFIG[table_name]
    placed_bboxes = []
    
    # 用于分类存储物体列表，以便后面控制显隐
    fixed_obj_parts = [] # 目标物体的所有部件
    other_obj_parts = [] # 其他干扰物的所有部件
    
    fixed_obj_root_ref = None # 用于计算矩阵的根节点

    # ==========================================
    # 4. Place objects
    # ==========================================
    for obj_glb in obj_glbs:
        old_objs = get_scene_objects_set()
        load_object(obj_glb)
        bpy.context.view_layer.update()
        
        # 获取当前物体导入的所有部件
        current_new_objs = get_new_objects(old_objs)
        root = find_root_object(current_new_objs)
        meshes = get_meshes_from_objects(current_new_objs)
        
        # 分类存储
        if obj_glb.endswith(fixed_obj_names + ".glb"):
            fixed_obj_parts.extend(current_new_objs)
            fixed_obj_root_ref = root
        else:
            other_obj_parts.extend(current_new_objs)

        placed = False
        
        for _ in range(100):
            # Sample XY
            if len(table_bbox_cfg) == 4:
                xmin, ymin, xmax, ymax = table_bbox_cfg
                x = random.uniform(xmin, xmax); y = random.uniform(ymin, ymax)
            else:
                R = table_bbox_cfg[0]
                r = random.uniform(0, R); theta = random.uniform(0, 2 * math.pi)
                x = r * math.cos(theta); y = r * math.sin(theta)

            root.location = (x, y, 0.0)
            bpy.context.view_layer.update()
            bbox_min, bbox_max = get_world_bbox_from_meshes(meshes)
            root.location.z = -bbox_min.z
            bpy.context.view_layer.update()
            bbox_min, bbox_max = get_world_bbox_from_meshes(meshes)
            bbox2d = ((bbox_min.x, bbox_min.y), (bbox_max.x, bbox_max.y))

            if not any(bbox_overlap_2d(bbox2d, b) for b in placed_bboxes):
                placed_bboxes.append(bbox2d)
                placed = True
                print(f"Placed {obj_glb} successfully.")
                break
    
        if not placed: raise RuntimeError(f"Failed to place {obj_glb}")

    if fixed_obj_root_ref is None:
        raise RuntimeError("Fixed object not found in scene!")

    # Calculate World2Object Matrix
    world2obj_matrix = np.array(fixed_obj_root_ref.matrix_world.inverted())
    # scale_x = np.linalg.norm(world2obj_matrix[0, :3])
    # scale_y = np.linalg.norm(world2obj_matrix[1, :3])
    # scale_z = np.linalg.norm(world2obj_matrix[2, :3])
    # world2obj_matrix[:3, :3] = world2obj_matrix[:3, :3] / np.array([scale_x, scale_y, scale_z])

    # 5. Lighting
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    env_tex = nodes.new(type="ShaderNodeTexEnvironment")
    hdri_bgs = os.listdir(args.hdrs_dir)
    hdri_bg = random.choice(hdri_bgs)
    env_tex.image = bpy.data.images.load(os.path.join(args.hdrs_dir, hdri_bg))
    bg = nodes.new(type="ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 1.0
    out = nodes.new(type="ShaderNodeOutputWorld")
    world.node_tree.links.new(env_tex.outputs["Color"], bg.inputs["Color"])
    world.node_tree.links.new(bg.outputs["Background"], out.inputs["Surface"])

    # 6. Camera Constraint
    empty = bpy.data.objects.new("Empty", None)
    bpy.context.scene.collection.objects.link(empty)
    cam_constraint.target = empty

    # 7. Render Loop
    scene_id = f"{table_name}__{'_'.join(obj_names)}"
    out_dir = Path(args.output_dir) / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    if args.camera_type == "random":
        distances = np.random.uniform(args.camera_dist_min, args.camera_dist_max, args.num_images)
        azimuths = np.linspace(0, 2*np.pi, args.num_images, endpoint=False)
        elevations = np.deg2rad(np.random.uniform(args.elevation_min, args.elevation_max, args.num_images))
    elif args.camera_type == "fixed":
        distances = np.full(args.num_images, args.camera_dist)
        azimuths = np.linspace(0, 2*np.pi, args.num_images, endpoint=False)
        elevations = np.deg2rad(np.full(args.num_images, args.elevation))
    
    cam_pts = az_el_to_points(azimuths, elevations) * distances[:, None]
    
    cam_poses = []
    obj2cam_poses = []

    for i in tqdm(range(args.num_images)):
        camera = set_camera_location(cam_pts[i])
        
        RT = get_3x4_RT_matrix_from_blender(camera)
        cam_poses.append(RT) 
        RT_4x4 = np.concatenate([RT, np.zeros((1, 4))], 0); RT_4x4[-1, -1] = 1
        obj2cam_matrix = RT_4x4 @ np.linalg.inv(world2obj_matrix)
        obj2cam_poses.append(obj2cam_matrix)

        # ==========================================
        # PASS 1: Render RGB Image
        # ==========================================
        # 1. 确保所有物体都可见
        for o in table_objs: o.hide_render = False
        for o in fixed_obj_parts: o.hide_render = False
        for o in other_obj_parts: o.hide_render = False
        
        # 2. 【关键修正】关闭透明背景，显示 HDRI 环境
        scene.render.film_transparent = False 
        
        # 3. 关闭材质覆盖，关闭合成器节点
        scene.view_layers[0].material_override = None
        xyz_out_node.mute = True
        mask_out_node.mute = True
        
        # 4. Render RGB
        render_path = out_dir / f"{i:03d}.png"
        scene.render.filepath = str(render_path)
        # with suppress_output(): # 调试时先注释掉
        bpy.ops.render.render(write_still=True)

        # ==========================================
        # PASS 2: Render XYZ Map & Mask
        # ==========================================
        # 1. 【关键修正】隐藏除了 fixed_obj 以外的所有物体
        # 显式隐藏 Table
        for o in table_objs: o.hide_render = True
        # 显式隐藏其他物体
        for o in other_obj_parts: o.hide_render = True
        # 显式显示目标物体
        for o in fixed_obj_parts: o.hide_render = False
        
        # 2. 【关键修正】开启透明背景，确保 Mask 的背景是 Alpha=0
        scene.render.film_transparent = True
        
        # 3. 开启材质覆盖，开启合成器节点
        scene.view_layers[0].material_override = coord_mat
        xyz_out_node.mute = False
        mask_out_node.mute = False
        
        # 设置输出路径
        xyz_out_node.base_path = str(out_dir)
        mask_out_node.base_path = str(out_dir)
        xyz_out_node.file_slots[0].path = f"{i:03d}_xyz"
        mask_out_node.file_slots[0].path = f"{i:03d}_mask"
        
        # 4. Render Data Pass
        scene.render.filepath = str(out_dir / "temp_data_pass.png") 
        # with suppress_output(): # 调试时先注释掉
        bpy.ops.render.render(write_still=False)

        # Clean up temp
        if os.path.exists(str(out_dir / "temp_data_pass.png")):
            os.remove(str(out_dir / "temp_data_pass.png"))

    # Save Meta Data
    cam_poses = np.stack(cam_poses)
    obj2cam_poses = np.stack(obj2cam_poses)
    K = get_calibration_matrix_K_from_blender(camera)

    for i in range(args.num_images):
        meta = dict(
            table=table_name,
            objects=obj_names,
            K=K,
            RT=cam_poses[i],
            obj2cam_poses=obj2cam_poses[i],
            azimuth=azimuths[i],
            elevation=elevations[i],
            distance=distances[i],
        )
        np.savez(out_dir / f"{i:03d}.npz", **meta)

    # Clean up materials
    bpy.data.materials.remove(coord_mat)

if __name__ == "__main__":
    save_images()