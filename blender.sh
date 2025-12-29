/data/user/rencw/blender-3.0.1-linux-x64/blender -b -P blender_script.py -- \
    --object_path glb_planes/plane_checker.glb \
    --output_dir ./views \
    --engine CYCLES \
    --num_images 12 \
    --camera_type fixed \
    --camera_dist 8 \
    --elevation 10 \
    --res_w 768 \
    --res_h 448 \
    --hdr_path hdri_bgs/brown_photostudio_02_4k.hdr \
    --output_dir ./views/test_tube_2racks_plane_brown_photostudio_02
    # --auto_offset True
    # --camera_dist 1.2