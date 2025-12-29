import argparse
import os
import os.path as osp

from easydict import EasyDict as edict

from vision3d.utils.io import ensure_dir

_C = edict()

# exp
_C.exp = edict()
_C.exp.name = osp.basename(osp.dirname(osp.realpath(__file__)))
_C.exp.working_dir = osp.dirname(osp.realpath(__file__))
_C.exp.output_dir = osp.join("./workspace/vision3d-output", _C.exp.name)
_C.exp.checkpoint_dir = osp.join(_C.exp.output_dir, "checkpoints")
_C.exp.log_dir = osp.join(_C.exp.output_dir, "logs")
_C.exp.event_dir = osp.join(_C.exp.output_dir, "events")
_C.exp.cache_dir = osp.join(_C.exp.output_dir, "cache")
_C.exp.result_dir = osp.join(_C.exp.output_dir, "results")
_C.exp.seed = 7351 
 
ensure_dir(_C.exp.output_dir)
ensure_dir(_C.exp.checkpoint_dir)
ensure_dir(_C.exp.log_dir)
ensure_dir(_C.exp.event_dir)
ensure_dir(_C.exp.cache_dir)
ensure_dir(_C.exp.result_dir)

# data
_C.data = edict()
_C.data.dataset_dir = "/zssd/dataset/lxj/nuscenes_corri2p_download/"

# train data
_C.train = edict()
_C.train.batch_size = 2
_C.train.num_workers = 8
_C.train.max_points = 40960
_C.train.scene_name = None

# test data
_C.test = edict()
_C.test.batch_size = 2
_C.test.num_workers = 8
_C.test.max_points = 40960
_C.test.scene_name = None

# trainer
_C.trainer = edict()
_C.trainer.max_epoch = 20
_C.trainer.grad_acc_steps = 1

# optim
_C.optimizer = edict()
_C.optimizer.type = "Adam"
_C.optimizer.lr = 1e-4
_C.optimizer.weight_decay = 1e-6

# scheduler
_C.scheduler = edict()
_C.scheduler.type = "Step"
_C.scheduler.gamma = 0.95
_C.scheduler.step_size = 1

# model - Global
_C.model = edict()
# model - image backbone
_C.model.image_backbone = edict()
_C.model.image_backbone.input_dim = 3
_C.model.image_backbone.output_dim = [512, 256, 128, 128]
_C.model.image_backbone.init_dim = 128
_C.model.image_backbone.dilation = 1

# model - point backbone
_C.model.point_backbone = edict()
_C.model.point_backbone.num_stages = 4
_C.model.point_backbone.base_voxel_size = 0.15
_C.model.point_backbone.kernel_size = 15
_C.model.point_backbone.kpconv_radius = 2.5
_C.model.point_backbone.kpconv_sigma = 2.0
_C.model.point_backbone.input_dim = 1
_C.model.point_backbone.init_dim = 64
_C.model.point_backbone.output_dim = [1024, 512, 256, 128]

def make_cfg():
    return _C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--link_output", dest="link_output", action="store_true", help="link output dir")
    args = parser.parse_args()
    return args


def main():
    cfg = make_cfg()
    args = parse_args()
    if args.link_output:
        os.symlink(cfg.output_dir, "output")


if __name__ == "__main__":
    main()
