CUDA_VISIBLE_DEVICES=0 python test.py --checkpoint=/ssd/lxj/release_code/ICL++/nuscenes/stage_1/workspace/vision3d-output/stage_1/checkpoints/checkpoint.pth
python eval_RTE_RRE.py
python eval_overlap_detection.py
