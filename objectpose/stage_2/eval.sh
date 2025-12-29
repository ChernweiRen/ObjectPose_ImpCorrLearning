CUDA_VISIBLE_DEVICES=0 python test.py --checkpoint=/ssd/lxj/release_code/ICL++/nuscenes/stage_2/workspace/vision3d-output/stage_2/checkpoints/checkpoint.pth
python eval_RRE_RTE_reg.py
python eval_RRE_RTE_pnp.py