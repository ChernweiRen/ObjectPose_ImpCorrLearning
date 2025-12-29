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

# sys.path.append("/data/user/rencw/ICL-I2PReg/")
# from blender_utils.vis_check import check_visibility


from contextlib import contextmanager

@contextmanager
def suppress_output():
    # 1. 打开空设备 (Linux/Mac是/dev/null, Windows是nul)
    null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
    
    # 2. 保存当前的标准输出/错误输出的文件描述符 (File Descriptors)
    save_fds = [os.dup(1), os.dup(2)]

    try:
        # 3. 将标准输出(1)和错误输出(2)重定向到空设备
        os.dup2(null_fds[0], 1)
        os.dup2(null_fds[1], 2)
        yield
    finally:
        # 4. 恢复原来的文件描述符
        os.dup2(save_fds[0], 1)
        os.dup2(save_fds[1], 2)
        
        # 5. 关闭临时文件句柄
        for fd in null_fds + save_fds:
            os.close(fd)



def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, pkl_path):
    # os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)


# the cfgs
TABLE_CONFIG = {
    "plane_checker": [-8., -8., 8., 8.],
    "plane_furn_cabinet": [-7.96, -3.46, 9.1, 3.46],
    "plane_furn": [7.32],
    "plane_gray": [-4.0, -4.0, 4.0, 4.0],
    "plane_wood": [-2.96, -3.98, 2.96, 3.98],
    "plane_table": [-4.6, -12.68, 4.6, 12.68],
}

TABLE_LIST = list(TABLE_CONFIG.keys())

# FIXED_OBJECT_LIST = [
#     "test_tube_rack",
# ]
fixed_obj_names = "test_tube_rack"

OBJECT_LIST = [
    "dropper_rack",
    "alarm_clock",
    "coffee_cup",
    "mass_chalice",
]

def get_all_children(obj):
    """Recursively get all children of an object (Blender 3.0 compatible)"""
    children = []
    for c in obj.children:
        children.append(c)
        children.extend(get_all_children(c))
    return children


# get the world bbox of an object
def get_world_bbox(obj):
    bbox_min = Vector((math.inf, math.inf, math.inf))
    bbox_max = Vector((-math.inf, -math.inf, -math.inf))

    meshes = []

    # Case 1: obj itself is mesh
    if obj.type == 'MESH':
        meshes.append(obj)

    # Case 2: recursive children (Blender 3.0 safe)
    for c in get_all_children(obj):
        if c.type == 'MESH':
            meshes.append(c)

    # Case 3: collection instance (Sketchfab / glTF)
    if obj.type == 'EMPTY' and obj.instance_type == 'COLLECTION':
        for c in obj.instance_collection.objects:
            if c.type == 'MESH':
                meshes.append(c)
            for cc in get_all_children(c):
                if cc.type == 'MESH':
                    meshes.append(cc)

    if not meshes:
        raise RuntimeError(f"No mesh found in object hierarchy: {obj.name}")

    for m in meshes:
        for v in m.bound_box:
            p = m.matrix_world @ Vector(v)
            bbox_min = Vector((
                min(bbox_min[0], p[0]),
                min(bbox_min[1], p[1]),
                min(bbox_min[2], p[2]),
            ))
            bbox_max = Vector((
                max(bbox_max[0], p[0]),
                max(bbox_max[1], p[1]),
                max(bbox_max[2], p[2]),
            ))

    return bbox_min, bbox_max


def bbox_overlap_2d(bbox_a, bbox_b):
    (ax_min, ay_min), (ax_max, ay_max) = bbox_a
    (bx_min, by_min), (bx_max, by_max) = bbox_b

    return not (
        ax_max <= bx_min or
        ax_min >= bx_max or
        ay_max <= by_min or
        ay_min >= by_max
    )

def get_scene_objects_set():
    return set(bpy.context.scene.objects)


def get_new_objects(old_set):
    return [o for o in bpy.context.scene.objects if o not in old_set]


def get_meshes_from_objects(objs):
    meshes = []
    for o in objs:
        if o.type == 'MESH':
            meshes.append(o)
        if o.type == 'EMPTY' and o.instance_type == 'COLLECTION':
            for c in o.instance_collection.objects:
                if c.type == 'MESH':
                    meshes.append(c)
    return meshes


def get_world_bbox_from_meshes(meshes):
    if not meshes:
        raise RuntimeError("No mesh found in imported object")

    bbox_min = Vector((math.inf, math.inf, math.inf))
    bbox_max = Vector((-math.inf, -math.inf, -math.inf))

    for m in meshes:
        for v in m.bound_box:
            p = m.matrix_world @ Vector(v)
            bbox_min.x = min(bbox_min.x, p.x)
            bbox_min.y = min(bbox_min.y, p.y)
            bbox_min.z = min(bbox_min.z, p.z)
            bbox_max.x = max(bbox_max.x, p.x)
            bbox_max.y = max(bbox_max.y, p.y)
            bbox_max.z = max(bbox_max.z, p.z)

    return bbox_min, bbox_max


def find_root_object(objs):
    for o in objs:
        if o.parent is None:
            return o
    return objs[0]



parser = argparse.ArgumentParser()
parser.add_argument("--object_path", type=str, required=True)
parser.add_argument("--plane_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])

# camera dist
parser.add_argument("--camera_dist", type=float, default=1.2)
parser.add_argument("--camera_dist_min", type=float, default=1.2)
parser.add_argument("--camera_dist_max", type=float, default=2.2)
# camera elevation
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

print('===================', args.engine, '===================')

context = bpy.context
scene = context.scene
render = scene.render

cam = scene.objects["Camera"]
# cam.location = (0, 1.2, 0)
cam.location = (0, 0, 0)
cam.data.lens = 35
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = args.res_w
render.resolution_y = args.res_h
render.resolution_percentage = 100

scene.cycles.device = "GPU"
# scene.cycles.samples = 128
scene.cycles.samples = 512 # increase to get better quality
scene.cycles.use_adaptive_sampling = True
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
# scene.cycles.filter_width = 0.01
scene.cycles.filter_width = 1.
scene.cycles.pixel_filter_type = 'GAUSSIAN'
scene.cycles.use_denoising = True
scene.cycles.denoiser = 'OPENIMAGEDENOISE'
# scene.cycles.denoiser = 'OPTIX' # if NVIDIA RTX GPU
# scene.render.film_transparent = True
scene.render.film_transparent = False

bpy.context.preferences.addons["cycles"].preferences.get_devices()
# Set the device_type
bpy.context.preferences.addons["cycles"].preferences.compute_device_type = args.device # or "OPENCL"
bpy.context.scene.cycles.tile_size = 8192

def create_debug_marker(location):
    # Create a small sphere
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=location)
    obj = bpy.context.active_object
    obj.name = "DEBUG_MARKER"
    
    # Give it a bright red color
    mat = bpy.data.materials.new(name="DebugRed")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes["Principled BSDF"].inputs[0].default_value = (1, 0, 0, 1) # 红色
    
    # This step makes it emit light, so it can be seen everywhere
    # nodes["Principled BSDF"].inputs[26].default_value = (1, 0, 0, 1) # Emission (高版本Blender可能是inputs[19]或[26]，如果报错可删掉这行)
    
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)



def az_el_to_points(azimuths, elevations):
    x = np.cos(azimuths)*np.cos(elevations)
    y = np.sin(azimuths)*np.cos(elevations)
    z = np.sin(elevations)
    return np.stack([x,y,z],-1) #

def set_camera_location(cam_pt):
    # from https://blender.stackexchange.com/questions/18530/
    x, y, z = cam_pt # sample_spherical(radius_min=1.5, radius_max=2.2, maxz=2.2, minz=-2.2)
    camera = bpy.data.objects["Camera"]
    camera.location = x, y, z

    return camera

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
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_u
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0  # only use rectangular pixels

    K = np.asarray(((alpha_u, skew, u_0),
                    (0, alpha_v, v_0),
                    (0, 0, 1)),np.float64)
    return K


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


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

# function from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
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

def normalize_scene(normalize_scene=False):
    bbox_min, bbox_max = scene_bbox()

    if normalize_scene:
        scale = 1 / max(bbox_max - bbox_min)
    else:
        scale = 1
        
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    if args.auto_offset:
        print("Enable Auto Offset.")
        offset = -(bbox_min + bbox_max) / 2
        for obj in scene_root_objects():
            obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

def save_images():

    os.makedirs(args.output_dir, exist_ok=True)

    # =====================
    # 1. Reset scene
    # =====================
    reset_scene()

    # =====================
    # 2. Randomly choose table & objects
    # =====================
    table_name = random.choice(TABLE_LIST)
    table_glb = os.path.join(args.plane_path, table_name + ".glb")


    # randomly choose from OBJECT_LIST
    num_objs = random.randint(0, 2)
    obj_names = random.sample(OBJECT_LIST, num_objs)
    obj_names = [fixed_obj_names] + obj_names
    obj_glbs = [os.path.join(args.object_path, n + ".glb") for n in obj_names]

    print(f"[Compose] Table: {table_name}")
    print(f"[Compose] Objects: {obj_names}")

    # =====================
    # 3. Load table
    # =====================
    load_object(table_glb)
    bpy.context.view_layer.update()
    
    table_bbox_cfg = TABLE_CONFIG[table_name]
    placed_bboxes = []
    
    # =====================
    # 4. Place objects ONE BY ONE
    # =====================
    for obj_glb in obj_glbs:
    
        # ---- import object ----
        old_objs = get_scene_objects_set()
        load_object(obj_glb)
        bpy.context.view_layer.update()
        new_objs = get_new_objects(old_objs)
    
        root = find_root_object(new_objs)
        meshes = get_meshes_from_objects(new_objs)
    
        placed = False
    
        for _ in range(100):
    
            # ---- sample XY ----
            if len(table_bbox_cfg) == 4:
                xmin, ymin, xmax, ymax = table_bbox_cfg
                x = random.uniform(xmin, xmax)
                y = random.uniform(ymin, ymax)
            else:
                R = table_bbox_cfg[0]
                r = random.uniform(0, R)
                theta = random.uniform(0, 2 * math.pi)
                x = r * math.cos(theta)
                y = r * math.sin(theta)
    
            # ---- move root to XY ----
            root.location = (x, y, 0.0)
            bpy.context.view_layer.update()

            # ---- compute bbox ----
            bbox_min, bbox_max = get_world_bbox_from_meshes(meshes)
    
            # ---- absolute Z align ----
            root.location.z = -bbox_min.z
            bpy.context.view_layer.update()
    
            # ---- recompute bbox ----
            bbox_min, bbox_max = get_world_bbox_from_meshes(meshes)
            bbox2d = ((bbox_min.x, bbox_min.y),
                      (bbox_max.x, bbox_max.y))
    
            # ---- collision check ----
            if not any(bbox_overlap_2d(bbox2d, b) for b in placed_bboxes):
                placed_bboxes.append(bbox2d)
                placed = True
                print(f"Placed {obj_glb} successfully.")
                break
    
        if not placed:
            raise RuntimeError(f"Failed to place {obj_glb}")
        
        if obj_glb.endswith(fixed_obj_names + ".glb"):
            # the world2object matrix
            world2obj_matrix = root.matrix_world.inverted()
            # to numpy
            world2obj_matrix = np.array(world2obj_matrix)

    # =====================
    # 5. World / HDR
    # =====================
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    env_tex = nodes.new(type="ShaderNodeTexEnvironment")
    # randomly choose a hdr
    hdri_bgs = os.listdir(args.hdrs_dir)
    hdri_bg = random.choice(hdri_bgs)
    env_tex.image = bpy.data.images.load(os.path.join(args.hdrs_dir, hdri_bg))

    bg = nodes.new(type="ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 1.0

    out = nodes.new(type="ShaderNodeOutputWorld")
    links.new(env_tex.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], out.inputs["Surface"])

    # =====================
    # 6. Camera target
    # =====================
    empty = bpy.data.objects.new("Empty", None)
    bpy.context.scene.collection.objects.link(empty)
    cam_constraint.target = empty

    # =====================
    # 7. Render loop
    # =====================
    scene_id = f"{table_name}__{'_'.join(obj_names)}"
    print(f"Scene ID: {scene_id}")
    out_dir = Path(args.output_dir) / scene_id
    print(f"Output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.camera_type == "random":
        distances = np.random.uniform(args.camera_dist_min, args.camera_dist_max, args.num_images)
        azimuths = np.linspace(0, 2*np.pi, args.num_images, endpoint=False)
        elevations = np.random.uniform(args.elevation_min, args.elevation_max, args.num_images)
        elevations = np.deg2rad(elevations)
        cam_pts = az_el_to_points(azimuths, elevations) * distances[:, None]
    elif args.camera_type == "fixed":
        distances = np.full(args.num_images, args.camera_dist)
    
        azimuths = np.linspace(0, 2*np.pi, args.num_images, endpoint=False)
        elevations = np.deg2rad(np.full(args.num_images, args.elevation))
    
        cam_pts = az_el_to_points(azimuths, elevations) * distances[:, None]
    else:
        raise ValueError(f"Invalid camera type: {args.camera_type}")
    
    cam_poses = []
    obj2cam_poses = []

    for i in tqdm(range(args.num_images)):

        camera = set_camera_location(cam_pts[i])
        RT = get_3x4_RT_matrix_from_blender(camera)
        cam_poses.append(RT) # world2cam
        
        # camera2object = world2object * inv(world2cam)
        # 3x4 -> 4x4
        RT = np.concatenate([RT, np.zeros((1, 4))], 0)
        RT[-1, -1] = 1

        obj2cam_matrix = RT @ np.linalg.inv(world2obj_matrix) # w2c @ o2w
        obj2cam_poses.append(obj2cam_matrix)

        render_path = out_dir / f"{i:03d}.png"
        bpy.context.scene.render.filepath = str(render_path)
        with suppress_output():
            bpy.ops.render.render(write_still=True)

    cam_poses = np.stack(cam_poses)
    obj2cam_poses = np.stack(obj2cam_poses)
    K = get_calibration_matrix_K_from_blender(camera)

    # save_pickle(
    #     dict(
    #         table=table_name,
    #         objects=obj_names,
    #         K=K,
    #         RT=cam_poses,
    #         obj2cam_poses=obj2cam_poses,
    #         azimuths=azimuths,
    #         elevations=elevations,
    #     ),
    #     out_dir / "meta.pkl"
    # )
    
    # meta = dict(
    #     table=table_name,
    #     objects=obj_names,
    #     K=K,
    #     RT=cam_poses,
    #     obj2cam_poses=obj2cam_poses,
    #     azimuths=azimuths,
    #     elevations=elevations,
    # )
    # np.savez(out_dir / "meta.npz", **meta)
    

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

if __name__ == "__main__":
    save_images()