# /data/user/rencw/blender-3.0.1-linux-x64/blender -b -P blender_script_random_depth_mask.py -- \
/data/user/rencw/blender-4.4.3-linux-x64/blender -b -P blender_script_random_depth_mask.py -- \
    --object_path glb_objs \
    --plane_path glb_planes \
    --output_dir ./views_tube \
    --engine CYCLES \
    --num_images 12 \
    --camera_type random \
    --camera_dist_min 16 \
    --camera_dist_max 32 \
    --elevation_min 10 \
    --elevation_max 45 \
    --res_w 640 \
    --res_h 480 \
    --hdrs_dir /data/user/rencw/ICL-I2PReg/hdri_bgs \
    # --auto_offset True
    # --camera_dist 1.2