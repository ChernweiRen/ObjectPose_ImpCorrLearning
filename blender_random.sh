/data/user/rencw/blender-3.0.1-linux-x64/blender -b -P blender_script_random.py -- \
    --object_path glb_objs \
    --plane_path glb_planes \
    --output_dir ./views \
    --engine CYCLES \
    --num_images 12 \
    --camera_type random \
    --camera_dist_min 16 \
    --camera_dist_max 24 \
    --elevation_min 10 \
    --elevation_max 30 \
    --res_w 768 \
    --res_h 448 \
    --hdrs_dir /data/user/rencw/ICL-I2PReg/hdri_bgs \
    # --auto_offset True
    # --camera_dist 1.2