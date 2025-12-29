import time
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from vision3d.ops import (
    create_meshgrid,
    render,
    apply_transform,
)
# isort: split
from fusion_module import OverlapEstimator, VertexPred
from image_backbone import ImageOverlap
from point_backbone import PointOverlap


class MATR2D3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.img_input_dim = cfg.model.image_backbone.output_dim
        self.pcd_input_dim  = cfg.model.point_backbone.output_dim
        
        self.img_overlap = ImageOverlap(
            cfg.model.image_backbone.input_dim,
            cfg.model.image_backbone.output_dim[-1],
            cfg.model.image_backbone.init_dim,
            dilation=cfg.model.image_backbone.dilation,
        )

        self.pcd_overlap = PointOverlap(
            cfg.model.point_backbone.input_dim,
            cfg.model.point_backbone.output_dim[-1],
            cfg.model.point_backbone.init_dim,
            cfg.model.point_backbone.kernel_size,
            cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_radius,
            cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_sigma,
        )

        self.overlap_pred = OverlapEstimator()
        self.vertex_pred = VertexPred()
    
    def vertex2RT(self, vertex):
        batch = vertex.shape[0]
        mid_dir = F.normalize(vertex[:, 2:], dim = -1)
        R = torch.zeros([batch, 3, 3]).cuda()
        for i in range(batch):
            R[i, 0, 0] = mid_dir[i, 1] 
            R[i, 0, 2] = - mid_dir[i, 0]
            R[i, 1, 1] = 1
            R[i, 2, 0] = mid_dir[i, 0] 
            R[i, 2, 2] = mid_dir[i, 1]

        ct = torch.zeros([batch, 1, 3]).cuda()
        ct[:, 0, 0] = vertex[:, 0]
        ct[:, 0, 2] = vertex[:, 1]
        t = - torch.matmul(ct, R.transpose(-1,-2)).squeeze(1)

        RT = torch.zeros([batch, 4, 4]).cuda()
        RT[:,:3,:3] = R
        RT[:,:3, 3] = t
        return RT

    def forward(self, data_dict):
        """
        data_dict:
            batch_size: int
            index: [B], is not used
            image: [B, H, W, 3], input image
            intrinsics: [B, 3, 3], camera intrinsics
            transform: [B, 4, 4], PnP transform matrix
            feats: [B, N, 32], point features
            points: [B, N, 3], point coordinates
            lengths: [B, N], point lengths
        output_dict:
            pred_vertex: [B, 2]
            gt_mask: [B, 1, H, W]
            pred_mask: [B, 1, H, W]
            mask_confidence: [B, 1, H, W]
            duration: float
        """
        torch.cuda.synchronize()
        start_time = time.time()

        output_dict = {}

        # 1. Unpack data
        batch_size = data_dict["batch_size"]
        # 1.1 Unpack 2D data
        index = data_dict["index"]
        image = data_dict["image"].detach()  # (B, H, W, 3), gray scaling_factor [B, 160, 512, 3]
        intrinsics = data_dict["intrinsics"].detach()  # (B, 3, 3)
        transform = torch.stack(data_dict["transform"], dim = 0).detach()

        img_h = image.shape[1]
        img_w = image.shape[2]

        image = image.permute(0, 3, 1, 2)
        
        img_feats_c = self.img_overlap(image)
        global_img_feats = img_feats_c.view(batch_size, self.img_input_dim[0], -1).mean(dim=-1)

        pcd_feats = data_dict["feats"].detach()
        pcd_points_list = data_dict["points"]   # [N, 3]    data_dict["lengths"] = list[tensor: shape = batchsize, tensor: shape = batchsize, tensor: shape = batchsize, tensor: shape = batchsize]
        pcd_lengths_list = data_dict["lengths"]
        pcd_feats_c = self.pcd_overlap(pcd_feats, data_dict)
        pcd_points_c = pcd_points_list[-1].detach()
        pcd_lengths_c = pcd_lengths_list[-1].detach()
        # 2.1 Overlap detection: in-frustum confidence prediction
        
        mask_confidence = self.overlap_pred(pcd_feats_c, global_img_feats, pcd_points_c / 10, pcd_lengths_c)

        # 2.2 Overlap detection: frustum pose regression
        pred_vertex = self.vertex_pred(mask_confidence.detach(), global_img_feats.detach(), pcd_feats_c.detach(), pcd_points_c / 10, pcd_lengths_c)
        coarse_transform = self.vertex2RT(pred_vertex).detach()

        gt_mask = torch.zeros_like(mask_confidence, dtype = torch.bool).cuda()
        pred_mask = torch.zeros_like(mask_confidence, dtype = torch.bool).cuda()
        start_idx = 0
        for i in range(batch_size):
            lengths = pcd_lengths_c[i]
            end_idx = start_idx + lengths
            gt_pcd_pixels, _ = render(pcd_points_c[start_idx:end_idx], intrinsics[i], extrinsics=transform[i], rounding=False)
            gt_mask[start_idx:end_idx] = ((gt_pcd_pixels[:, 0] >= 0) & (gt_pcd_pixels[:, 0] < img_h) & (gt_pcd_pixels[:, 1] >= 0) & (gt_pcd_pixels[:, 1] < img_w))
            pred_pcd_pixels, _ = render(pcd_points_c[start_idx:end_idx], intrinsics[i], extrinsics=coarse_transform[i], rounding=False)
            pred_mask[start_idx:end_idx] = ((pred_pcd_pixels[:, 0] >= 0) & (pred_pcd_pixels[:, 0] < img_h) & (pred_pcd_pixels[:, 1] >= 0) & (pred_pcd_pixels[:, 1] < img_w))
            start_idx = end_idx
            
        output_dict["pred_vertex"] = pred_vertex
        output_dict["gt_mask"] = gt_mask
        output_dict["pred_mask"] = pred_mask
        output_dict["mask_confidence"] = mask_confidence

        # this_folder = os.getcwd()
        # store_dir = this_folder + "/workspace/point_classification/"
        # cls_record_dir = this_folder + "/workspace/cls_record.txt"
        # f = open(cls_record_dir,"a") 
        # if not os.path.exists(store_dir):
        #     os.makedirs(store_dir)
        # start_idx = 0
        # pcd_mask = gt_mask
        # for i in range(batch_size):
        #     lengths = pcd_lengths_c[i]
        #     end_idx = start_idx + lengths
        #     idx = index[i]
        #     before_store_path = store_dir  + 'before_' + str(idx) + '.npy'
        #     after_store_path = store_dir  + 'after_' + str(idx) + '.npy'
        #     gt_store_path = store_dir  + 'gt_' + str(idx) + '.npy'
        #     points = pcd_points_c[start_idx:end_idx].cpu().numpy()
        #     before_mask = (mask_confidence[start_idx:end_idx] > 0).cpu().numpy()
        #     after_mask = pred_mask[start_idx:end_idx].cpu().numpy()
        #     gt_mask = pcd_mask[start_idx:end_idx].cpu().numpy()
            
        #     before_precise = (before_mask & gt_mask).sum(-1) / before_mask.sum(-1)
        #     before_recall = (before_mask & gt_mask).sum(-1) / gt_mask.sum(-1)

        #     after_precise = (after_mask & gt_mask).sum(-1) / after_mask.sum(-1)
        #     after_recall = (after_mask & gt_mask).sum(-1) / gt_mask.sum(-1)

        #     if after_precise + after_recall - before_precise - before_recall > 0.3:
        #         before_colors = np.zeros_like(points)
        #         before_colors[(before_mask & gt_mask)] = [0,255,0]
        #         before_colors[(~before_mask & gt_mask)] = [0,0,255]
        #         before_colors[(before_mask & ~gt_mask)] = [255,0,0]
        #         before_pcd = np.concatenate([points, before_colors], axis = 1)
        #         np.save(before_store_path, before_pcd)

        #         after_colors = np.zeros_like(points)
        #         after_colors[(after_mask & gt_mask)] = [0,255,0]
        #         after_colors[(~after_mask & gt_mask)] = [0,0,255]
        #         after_colors[(after_mask & ~gt_mask)] = [255,0,0]
        #         after_pcd = np.concatenate([points, after_colors], axis = 1)
        #         np.save(after_store_path, after_pcd)

        #         gt_colors = np.zeros_like(points)
        #         gt_colors[gt_mask] = [0,255,0]
        #         gt_pcd = np.concatenate([points, gt_colors], axis = 1)
        #         np.save(gt_store_path, gt_pcd)
        #         f.write(str(idx) + ' ' + str(before_precise) + ' ' + str(before_recall) + ' ' + str(after_precise) + ' ' + str(after_recall) + '\n')
        # f.close()

        torch.cuda.synchronize()
        duration = time.time() - start_time
        output_dict["duration"] = duration

        return output_dict

def create_model(cfg):
    model = MATR2D3D(cfg)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == "__main__":
    main()
