"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2

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

from mathutils import Vector, Matrix
import numpy as np

import bpy
from mathutils import Vector
import pickle

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, pkl_path):
    # os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

parser = argparse.ArgumentParser()
parser.add_argument("--object_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])
parser.add_argument("--camera_type", type=str, default='even')
parser.add_argument("--camera_dist", type=float, default=1.2)
parser.add_argument("--num_images", type=int, default=16)
parser.add_argument("--elevation", type=float, default=30)
parser.add_argument("--elevation_start", type=float, default=-10)
parser.add_argument("--elevation_end", type=float, default=40)
parser.add_argument("--res_w", type=int, default=256)
parser.add_argument("--res_h", type=int, default=256)
parser.add_argument("--hdr_path", type=str, default="/data/user/rencw/hdr/myhdr.hdr")
parser.add_argument("--auto_offset", type=bool, default=False)
parser.add_argument("--normalize_scene", type=bool, default=False)
parser.add_argument("--device", type=str, default='CUDA')


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
    # 创建一个小球
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=location)
    obj = bpy.context.active_object
    obj.name = "DEBUG_MARKER"
    
    # 给它上个鲜艳的红色
    mat = bpy.data.materials.new(name="DebugRed")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes["Principled BSDF"].inputs[0].default_value = (1, 0, 0, 1) # 红色
    
    # 这一步是为了让它自发光，不受光照影响，哪里都能看见
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

def save_images(object_file: str) -> None:
    object_uid = os.path.basename(object_file).split(".")[0]
    os.makedirs(args.output_dir, exist_ok=True)

    reset_scene()
    # load the object
    load_object(object_file)
    # object_uid = os.path.basename(object_file).split(".")[0]
    normalize_scene()

    # # DEBUG for checking the origin of the object
    # print("\n" + "="*40) # DEBUG
    # print("DEBUG: 正在检查所有物体的【原点 Origin】位置")
    
    # for obj in bpy.context.scene.objects:
    #     # 只检查网格模型 (Mesh)，忽略灯光和相机
    #     if obj.type == 'MESH':
    #         # 获取世界坐标系下的绝对位置
    #         origin_loc = obj.matrix_world.translation
            
    #         print(f"物体名称: {obj.name}")
    #         print(f"  -> 原点坐标 (Origin): {origin_loc}")
            
    #         # 这里的坐标如果不接近 (0,0,0)，说明物体本身就没有归零
            
    #         # === 可视化：在原点生成一个蓝色小球 ===
    #         # (需要保留之前定义的 create_debug_marker 函数，稍微改个颜色即可)
    #         # 这里直接简单生成一个球
    #         bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, location=origin_loc)
    #         debug_sphere = bpy.context.active_object
    #         debug_sphere.name = f"Origin_Marker_{obj.name}"
            
    # print("="*40 + "\n")


    # ---- Add HDR Environment Lighting ----
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    for n in nodes:
        nodes.remove(n)

    env_tex = nodes.new(type="ShaderNodeTexEnvironment")
    env_tex.image = bpy.data.images.load(args.hdr_path)

    bg = nodes.new(type="ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 1.2

    out = nodes.new(type="ShaderNodeOutputWorld")

    links.new(env_tex.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], out.inputs["Surface"])

    # # DEBUG
    # bbox_min, bbox_max = scene_bbox()
    # target_point = (bbox_min + bbox_max) / 2
    # print(f"DEBUG: 在坐标 {target_point} 处生成标记球")
    # create_debug_marker(target_point + Vector((0, 0, 0.5)))


    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    world_tree = bpy.context.scene.world.node_tree
    back_node = world_tree.nodes['Background']
    env_light = 0.5
    back_node.inputs['Color'].default_value = Vector([env_light, env_light, env_light, 1.0])
    back_node.inputs['Strength'].default_value = 1.0

    distances = np.asarray([args.camera_dist for _ in range(args.num_images)])
    if args.camera_type=='fixed':
        azimuths = (np.arange(args.num_images)/args.num_images*np.pi*2).astype(np.float64)
        elevations = np.deg2rad(np.asarray([args.elevation] * args.num_images).astype(np.float64))
    elif args.camera_type=='random':
        azimuths = (np.arange(args.num_images) / args.num_images * np.pi * 2).astype(np.float64)
        elevations = np.random.uniform(args.elevation_start, args.elevation_end, args.num_images)
        elevations = np.deg2rad(elevations)
    else:
        raise NotImplementedError

    cam_pts = az_el_to_points(azimuths, elevations) * distances[:,None]
    cam_poses = []
    (Path(args.output_dir) / object_uid).mkdir(exist_ok=True, parents=True)
    for i in range(args.num_images):
        # set camera
        camera = set_camera_location(cam_pts[i])
        RT = get_3x4_RT_matrix_from_blender(camera)
        cam_poses.append(RT)

        render_path = os.path.join(args.output_dir, object_uid, f"{i:03d}.png")
        if os.path.exists(render_path): continue
        scene.render.filepath = os.path.abspath(render_path)
        bpy.ops.render.render(write_still=True)

    if args.camera_type=='random':
        K = get_calibration_matrix_K_from_blender(camera)
        cam_poses = np.stack(cam_poses, 0)
        save_pickle([K, azimuths, elevations, distances, cam_poses], os.path.join(args.output_dir, object_uid, "meta.pkl"))

if __name__ == "__main__":
    save_images(args.object_path)