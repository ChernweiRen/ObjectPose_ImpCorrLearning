from plyfile import PlyData

ply = PlyData.read("/data/user/rencw/ICL-I2PReg/ply_objs/tube_thick_vA_Blender.ply")
faces = ply['face'].data

bad = [f for f in faces if len(f[0]) != 3]
print("non-triangle faces:", len(bad))