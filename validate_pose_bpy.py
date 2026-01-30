import bpy
import sys
import os
import argparse
import numpy as np
from mathutils import Matrix
import math

# ------------------------------------------------
# argparse for Blender
# ------------------------------------------------
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

parser = argparse.ArgumentParser()
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--out", type=str, default="render.png")
args = parser.parse_args(argv)

# ------------------------------------------------
# Clean scene
# ------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)

# ------------------------------------------------
# Import mesh
# ------------------------------------------------
# mesh_path = "/data/user/rencw/ICL-I2PReg/ply_objs/test_tube_rack.ply"
# mesh_path = "/data/user/rencw/ICL-I2PReg/ply_objs/test_tube_rack_vB_GLTF.ply"
# mesh_path = "/data/user/rencw/ICL-I2PReg/ply_objs/test_tube_rack_vC_NegZ.ply"
# mesh_path = "/data/user/rencw/ICL-I2PReg/glb_objs/test_tube_rack.glb"
# mesh_path = "/data/user/rencw/ICL-I2PReg/ply_objs/tube_vC_NegZ.ply"
# mesh_path = "/data/user/rencw/ICL-I2PReg/ply_objs/tube_vA_Blender.ply"
mesh_path = "/data/user/rencw/ICL-I2PReg/ply_objs/tube_thick_vA_Blender.ply"
# mesh_path = "/data/user/rencw/ICL-I2PReg/ply_objs/tube_thick_vB_GLTF.ply"
# mesh_path = "/data/user/rencw/ICL-I2PReg/ply_objs/tube_thick_vC_NegZ.ply"
# mesh_path = "/data/user/rencw/ICL-I2PReg/ply_objs/tube_thick_vD_Swap.ply"
ext = os.path.splitext(mesh_path)[1]

global_scale = 20.
# global_scale = 1.0

if ext == ".ply":
    # bpy.ops.import_mesh.ply(filepath=mesh_path)
    bpy.ops.wm.ply_import(filepath=mesh_path)
elif ext == ".obj":
    bpy.ops.import_scene.obj(filepath=mesh_path)
elif ext in [".glb", ".gltf"]:
    bpy.ops.import_scene.gltf(filepath=mesh_path)
else:
    raise ValueError("Unsupported mesh format")

obj = bpy.context.selected_objects[0]
print(obj.scale)
# obj.scale = (0.001, 0.001, 0.001)
obj.scale = (0.001/global_scale, 0.001/global_scale, 0.001/global_scale)
# location
print(obj.location)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# ------------------------------------------------
# Load pose (object-to-camera)
# ------------------------------------------------
# npz_data = np.load("/data/user/rencw/ICL-I2PReg/views/plane_furn__test_tube_rack_mass_chalice/007.npz")
# npz_data = np.load("/data/user/rencw/ICL-I2PReg/views_tube/plane_furn__tube/007.npz")
npz_data = np.load("/data/user/rencw/ICL-I2PReg/views_tube/plane_table.glb__tube_thick.blend/002.npz")
RT = npz_data["obj2cam_poses"]
RT_copy = RT.copy()
# unscale the rotation
scale_x = np.linalg.norm(RT_copy[0, :3])
scale_y = np.linalg.norm(RT_copy[1, :3])
scale_z = np.linalg.norm(RT_copy[2, :3])
print(scale_x, scale_y, scale_z)
# if mesh_path.endswith("ply"):
#     RT_copy[:3, :3] = RT_copy[:3, :3] / np.array([scale_x, scale_y, scale_z])
RT_copy[:3, :3] = RT_copy[:3, :3] / np.array([scale_x, scale_y, scale_z])
RT_copy[:3, 3] = RT_copy[:3, 3] / global_scale
RT = RT_copy
# RT = np.linalg.inv(RT)
print(RT)
if RT.shape == (3, 4):
    RT = np.vstack([RT, [0, 0, 0, 1]])

# OpenCV → Blender camera coords
cv2blender = np.array([
    [1,  0,  0,  0],
    [0, -1,  0,  0],
    [0,  0, -1,  0],
    [0,  0,  0,  1]
])

RT_blender = cv2blender @ RT
# RT_blender = RT

# if mesh_path.endswith("ply"):
#     rot_correction = Matrix.Rotation(math.radians(-90), 4, 'X')
#     fixed_matrix = Matrix(RT_blender) @ rot_correction
#     obj.matrix_world = fixed_matrix
# else:
#     obj.matrix_world = Matrix(RT_blender)
obj.matrix_world = Matrix(RT_blender)

# ------------------------------------------------
# Camera
# ------------------------------------------------
cam = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam)
bpy.context.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

# Camera intrinsics
K = npz_data["K"]
print(K)
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

W, H = args.width, args.height

cam.type = "PERSP"
cam.lens_unit = "FOV"

cam.angle_x = 2 * np.arctan(W / (2 * fx))
cam.angle_y = 2 * np.arctan(H / (2 * fy))

cam.shift_x = (cx - W / 2) / W
cam.shift_y = (H / 2 - cy) / H

cam.clip_start = 0.001
cam.clip_end = 100.0

cam_obj.location = (0, 0, 0)
cam_obj.rotation_euler = (0, 0, 0)

# ------------------------------------------------
# Light
# ------------------------------------------------
light = bpy.data.lights.new(name="Light", type="SUN")
light_obj = bpy.data.objects.new(name="Light", object_data=light)
light_obj.rotation_euler = (0.7, 0.0, 0.7)
bpy.context.collection.objects.link(light_obj)

# ------------------------------------------------
# Render settings
# ------------------------------------------------
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "GPU"
scene.render.resolution_x = W
scene.render.resolution_y = H
scene.render.film_transparent = True

scene.render.filepath = args.out
scene.render.image_settings.file_format = "PNG"

# ------------------------------------------------
# Render
# ------------------------------------------------
bpy.ops.render.render(write_still=True)

print(f"[OK] Render saved to {args.out}")