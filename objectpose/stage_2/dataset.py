from objectpose_dataset import objectPosePairDataset
from vision3d.utils.collate import GraphPyramid2D3DRegistrationCollateFn
from vision3d.utils.dataloader import build_dataloader, calibrate_neighbors_pack_mode
import options as opt

def train_valid_data_loader(cfg):
    train_dataset = objectPosePairDataset(root_path="/data/user/rencw/ICL-I2PReg/views",
                                    mode='train',
                                    subfolder=["plane_checker__test_tube_rack_alarm_clock_dropper_rack"],
                                    obj_glb_path="/data/user/rencw/ICL-I2PReg/glb_objs/test_tube_rack.glb",
                                    img_size=(448, 768),
                                    downsample_ratio=16
                                    )

    neighbor_limits = calibrate_neighbors_pack_mode(
        train_dataset,
        GraphPyramid2D3DRegistrationCollateFn,
        cfg.model.point_backbone.num_stages,
        cfg.model.point_backbone.base_voxel_size,
        cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_radius,
    )

    collate_fn = GraphPyramid2D3DRegistrationCollateFn(
        cfg.model.point_backbone.num_stages,
        cfg.model.point_backbone.base_voxel_size,
        cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_radius,
        neighbor_limits,
    )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last = True,
    )

    valid_dataset = objectPosePairDataset(root_path="/data/user/rencw/ICL-I2PReg/views",
                                    mode='train',
                                    subfolder=["plane_checker__test_tube_rack_alarm_clock_dropper_rack"],
                                    obj_glb_path="/data/user/rencw/ICL-I2PReg/glb_objs/test_tube_rack.glb",
                                    img_size=(448, 768),
                                    downsample_ratio=16
                                    )

    valid_loader = build_dataloader(
        valid_dataset,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last = True,
    )

    return train_loader, valid_loader, neighbor_limits


def test_data_loader(cfg):
    train_dataset = objectPosePairDataset(root_path="/data/user/rencw/ICL-I2PReg/views",
                                    mode='train',
                                    subfolder=["plane_checker__test_tube_rack_alarm_clock_dropper_rack"],
                                    obj_glb_path="/data/user/rencw/ICL-I2PReg/glb_objs/test_tube_rack.glb",
                                    img_size=(448, 768),
                                    downsample_ratio=16
                                    )

    neighbor_limits = calibrate_neighbors_pack_mode(
        train_dataset,
        GraphPyramid2D3DRegistrationCollateFn,
        cfg.model.point_backbone.num_stages,
        cfg.model.point_backbone.base_voxel_size,
        cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_radius,
    )

    collate_fn = GraphPyramid2D3DRegistrationCollateFn(
        cfg.model.point_backbone.num_stages,
        cfg.model.point_backbone.base_voxel_size,
        cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_radius,
        neighbor_limits,
    )

    test_dataset = objectPosePairDataset(root_path="/data/user/rencw/ICL-I2PReg/views",
                                    mode='train',
                                    subfolder=["plane_checker__test_tube_rack_alarm_clock_dropper_rack"],
                                    obj_glb_path="/data/user/rencw/ICL-I2PReg/glb_objs/test_tube_rack.glb",
                                    img_size=(448, 768),
                                    downsample_ratio=16
                                    )

    test_loader = build_dataloader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last = True,
    )

    return test_loader, neighbor_limits