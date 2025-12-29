import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


def is_object_in_frame(scene, camera, obj):
    """
    检测物体是否在摄像机视野内（不考虑遮挡）。
    """
    # 确保依赖图是最新的，获得物体的最终变换
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    
    # 获取物体的包围盒顶点（世界坐标）
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    
    in_view = False
    
    for corner in bbox_corners:
        # 转换到摄像机坐标系: (x, y, z_depth)
        # x, y 范围是 0.0 到 1.0 代表在画面内
        # z 代表深度，必须 > 0 才是摄像机前方
        coords = world_to_camera_view(scene, camera, corner)
        
        if 0.0 <= coords.x <= 1.0 and 0.0 <= coords.y <= 1.0 and coords.z > 0.0:
            in_view = True
            break  # 只要有一个角在画面内，就判定为在画面内
            
    return in_view

# === 用法示例 ===
# camera = scene.objects["Camera"]
# obj = scene.objects["my_object"]
# if is_object_in_frame(scene, camera, obj):
#     print(f"{obj.name} 在画面范围内")



def is_object_visible_by_raycast(scene, camera, obj):
    """
    检测物体是否可见（考虑遮挡）。
    从摄像机位置向物体几何中心发射射线。
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    
    # 1. 获取射线的起点（摄像机位置）
    origin = camera.matrix_world.translation
    
    # 2. 获取射线的终点（物体中心）
    # 注意：这里用简单的包围盒中心，对于空心的甜甜圈形状可能不准确，但对一般物体够用
    # 更严谨的做法是随机采样物体表面的几个顶点
    bbox_center = sum([Vector(b) for b in obj.bound_box], Vector()) / 8
    target = obj.matrix_world @ bbox_center
    
    # 3. 计算方向向量
    direction = (target - origin).normalized()
    distance = (target - origin).length
    
    # 4. 发射射线
    # ray_cast 返回: result, location, normal, index, object, matrix
    is_hit, _, _, _, hit_obj, _ = scene.ray_cast(depsgraph, origin, direction, distance=distance)
    
    # 5. 判断逻辑
    if not is_hit:
        # 如果射线没打中任何东西（但在距离内），说明可能直接穿过去了或者出错了
        # 这里视情况而定，通常如果打中了才是遮挡
        return True 
        
    if hit_obj.name == obj.name:
        return True  # 直接打中了目标物体，可见
    else:
        # print(f"被 {hit_obj.name} 挡住了")
        return False # 打中了别的物体（遮挡物）

# === 用法示例 ===
# if is_object_visible_by_raycast(scene, camera, obj):
#     print(f"{obj.name} 没有被遮挡")



def check_visibility(scene, camera, obj):
    # 1. 视锥体检查 (Frustum Check)
    in_frame = False
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    for corner in corners:
        co = world_to_camera_view(scene, camera, corner)
        if 0.0 <= co.x <= 1.0 and 0.0 <= co.y <= 1.0 and co.z > 0:
            in_frame = True
            break
    
    if not in_frame:
        return False, "Out of Frame"

    # 2. 遮挡检查 (Occlusion Check) - 简易版，只查中心点
    # 为了更稳，可以把 target 换成 corners 里的随机一点
    depsgraph = bpy.context.evaluated_depsgraph_get()
    origin = camera.matrix_world.translation
    
    # 计算物体世界坐标中心
    bbox_center = sum([Vector(b) for b in obj.bound_box], Vector()) / 8
    target = obj.matrix_world @ bbox_center
    
    direction = (target - origin).normalized()
    dist = (target - origin).length
    
    # 发射射线
    is_hit, loc, normal, idx, hit_obj, mat = scene.ray_cast(depsgraph, origin, direction, distance=dist - 0.01) # 稍微减一点距离防止浮点误差
    
    if is_hit and hit_obj.name != obj.name:
        # 如果打中的不是自己，且不是自己的子物体（视情况），则被遮挡
        return False, f"Occluded by {hit_obj.name}"
        
    return True, "Visible"