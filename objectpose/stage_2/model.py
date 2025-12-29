import time
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from vision3d.models.geotransformer import SuperPointMatchingMutualTopk, SuperPointProposalGenerator
from vision3d.ops import (
    create_meshgrid,
    render,
    apply_transform,
    get_transform_from_rotation_translation,
)
# isort: split
from fusion_module import CrossModalFusionModule, FeatureFusion, OverlapEstimator, VertexPred
from image_backbone import ImageBackbone, ImageOverlap
from point_backbone import PointBackbone, PointOverlap

class MATR2D3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.decoder_input_dim = cfg.model.transformer.input_dim
        self.decoder_output_dim = cfg.model.transformer.output_dim
        self.img_input_dim = cfg.model.image_backbone.output_dim
        self.pcd_input_dim  = cfg.model.point_backbone.output_dim
        self.num_queries = cfg.model.transformer.num_queries
        self.num_stages = cfg.model.point_backbone.num_stages

        self.img_radius = cfg.model.transformer.img_radius_thresold
        self.pcd_radius = cfg.model.transformer.pcd_radius_thresold

        self.img_backbone = ImageBackbone(
            cfg.model.image_backbone.input_dim,
            cfg.model.image_backbone.output_dim[-1],
            cfg.model.image_backbone.init_dim,
            dilation=cfg.model.image_backbone.dilation,
        )

        self.pcd_backbone = PointBackbone(
            cfg.model.point_backbone.input_dim,
            cfg.model.point_backbone.output_dim[-1],
            cfg.model.point_backbone.init_dim,
            cfg.model.point_backbone.kernel_size,
            cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_radius,
            cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_sigma,
        )

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

        self.query_embed = nn.Embedding(self.num_queries, self.decoder_input_dim[0])

        self.decoder = nn.ModuleList()
        self.fusion_block =  nn.ModuleList()
        for i in range(self.num_stages):
            self.decoder.append(
                CrossModalFusionModule(
                    self.img_input_dim[i],
                    self.pcd_input_dim[i],
                    self.decoder_input_dim[i],
                    self.decoder_output_dim[i],

                )
            )
            self.fusion_block.append(FeatureFusion(self.decoder_input_dim[i]))
        
    def align_pcd_data(self, pcd_feats, pcd_points, pcd_lengths):
        B = pcd_lengths.shape[0]
        C = pcd_feats.shape[-1]
        max_len = max(pcd_lengths)

        aligned_pcd_feats = torch.zeros(B, max_len, C).cuda()
        aligned_pcd_points = torch.zeros(B, max_len, 3).cuda().detach()
        align_mask = torch.ones(B, max_len, dtype=torch.bool).cuda().detach()
        
        start_idx = 0

        for i in range(B):
            lengths = pcd_lengths[i]
            end_idx = start_idx + lengths
            aligned_pcd_feats[i, :lengths] = pcd_feats[start_idx:end_idx]
            aligned_pcd_points[i, :lengths] = pcd_points[start_idx:end_idx]
            align_mask[i, :lengths] = False
            start_idx = end_idx
        return aligned_pcd_feats, aligned_pcd_points, align_mask
    
    def vertex2RT(self, vertex):
        batch = vertex.shape[0]
        mid_dir = vertex[:, 2:]
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

        RT = torch.zeros([batch, 4, 4]).cuda().detach()
        RT[:,:3,:3] = R
        RT[:,:3, 3] = t
        return RT

    def reverse_proj(self, pixels, intrinsics):
        focal_x = intrinsics[..., 0, 0].unsqueeze(-1)  # (1) or (B, 1)
        focal_y = intrinsics[..., 1, 1].unsqueeze(-1)  # (1) or (B, 1)
        center_x = intrinsics[..., 0, 2].unsqueeze(-1)  # (1) or (B, 1)
        center_y = intrinsics[..., 1, 2].unsqueeze(-1)  # (1) or (B, 1)

        h_coords = pixels[..., 0]
        w_coords = pixels[..., 1]

        x_coords = (w_coords - center_x) / focal_x
        y_coords = (h_coords - center_y) / focal_y

        normed_pixels = torch.stack([y_coords, x_coords], dim=-1)
        return normed_pixels

    def RT4D2RT(self, RT4D):
        batch = RT4D.shape[0]
        mid_dir = RT4D[:, :2]
        R = torch.zeros([batch, 3, 3]).cuda()
        for i in range(batch):
            R[i, 0, 0] = mid_dir[i, 1] 
            R[i, 0, 2] = - mid_dir[i, 0]
            R[i, 1, 1] = 1
            R[i, 2, 0] = mid_dir[i, 0] 
            R[i, 2, 2] = mid_dir[i, 1]

        t = torch.zeros([batch, 3]).cuda()
        t[:, 0] = RT4D[:, 2]
        t[:, 2] = RT4D[:, 3]

        RT = torch.zeros([batch, 4, 4]).cuda().detach()
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
        # 1.1 Unpack 2D data
        batch_size = data_dict["batch_size"]
        index = data_dict["index"]
        image = data_dict["image"].detach()  # (B, H, W, 3), gray scaling_factor [B, 160, 512, 3]
        intrinsics = data_dict["intrinsics"].detach()  # (B, 3, 3)
        transform = torch.stack(data_dict["transform"], dim = 0).detach()

        img_h = image.shape[1]
        img_w = image.shape[2]

        img_pixels_list = []
        for i in range(self.num_stages):
            img_pixels_list.append(create_meshgrid(img_h // 2**i, img_w // 2**i, normalized=False, flatten=True, centering = True).unsqueeze(0).repeat(batch_size, 1, 1) * 2**i)
        img_pixels_list.reverse()
        output_dict["img_pixels_list"] = img_pixels_list
        
        image = image.permute(0, 3, 1, 2)
        img_feats_list = self.img_backbone(image)   # size: [1, 1/2, 1/4, 1/8]   channel: [image_backbone.output_dim, image_backbone.init_dim * 1, image_backbone.init_dim * 2, image_backbone.init_dim * 4]
        img_feats_c = self.img_overlap(image)
        global_img_feats = img_feats_c.view(batch_size, self.img_input_dim[0], -1).mean(dim=-1)
        # 1.2 Unpack 3D data
        pcd_feats = data_dict["feats"].detach()

        pcd_points_list = data_dict["points"]   # [N, 3]    data_dict["lengths"] = list[tensor: shape = batchsize, tensor: shape = batchsize, tensor: shape = batchsize, tensor: shape = batchsize]
        pcd_lengths_list = data_dict["lengths"]
        pcd_feats_list = self.pcd_backbone(pcd_feats, data_dict)
        pcd_feats_c = self.pcd_overlap(pcd_feats, data_dict)
        pcd_points_list.reverse()
        pcd_lengths_list.reverse()

        # 2.1 Overlap detection: in-frustum confidence prediction
        mask_confidence = self.overlap_pred(pcd_feats_c, global_img_feats, pcd_points_list[0] / 10, pcd_lengths_list[0])

        # 2.2 Overlap detection: frustum pose regression
        pred_vertex = self.vertex_pred(mask_confidence, global_img_feats, pcd_feats_c, pcd_points_list[0] / 10, pcd_lengths_list[0])
        coarse_transform = self.vertex2RT(pred_vertex).detach()

        # 3.1 2D-3D correspondence generation
        img_keypoint_pixels_list = []
        pcd_keypoint_points_list = []
        RT_estimate_list = [coarse_transform]
        RT4D_estimate_list = []
        query = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        for i in range(self.num_stages):
            img_feats = img_feats_list[i].view(batch_size, self.img_input_dim[i], -1).transpose(1, 2)

            pcd_feats = pcd_feats_list[i]
            pcd_points = pcd_points_list[i].detach()
            pcd_lengths = pcd_lengths_list[i].detach()
            aligned_pcd_feats, aligned_pcd_points, align_mask = self.align_pcd_data(pcd_feats, pcd_points, pcd_lengths) 

            if i != 0:
                img_dist = torch.norm(img_keypoint_pixels.unsqueeze(2) - img_pixels_list[i].unsqueeze(1), dim = -1)
                pcd_dist = torch.norm(pcd_keypoint_points.unsqueeze(2) - aligned_pcd_points.unsqueeze(1), dim = -1)

                img_qk_masks = (img_dist > self.img_radius[i])
                pcd_qk_masks = (pcd_dist > self.pcd_radius[i])
                pcd_qk_masks = (pcd_qk_masks | align_mask.unsqueeze(1).repeat(1, self.num_queries, 1))

            else:
                aligned_pcd_pixels, _ = render(aligned_pcd_points, intrinsics, extrinsics=RT_estimate_list[-1], rounding=False)
                in_img_mask = ((aligned_pcd_pixels[:, :, 0] >= 0) & (aligned_pcd_pixels[:, :, 0] < img_h) & (aligned_pcd_pixels[:, :, 1] >= 0) & (aligned_pcd_pixels[:, :, 1] < img_w))
                align_mask = (align_mask | ~in_img_mask)
                img_qk_masks = None
                pcd_qk_masks = align_mask.unsqueeze(1).repeat(1, self.num_queries, 1)

            query_list, img_tokens, pcd_tokens = self.decoder[i](
                query,
                img_feats,
                aligned_pcd_feats,
                img_masks = img_qk_masks,
                pcd_masks = pcd_qk_masks,
            )

            query_img_feats = query_list[0]
            query_pcd_feats = query_list[1]
            query = query_list[2]

            img_keypoint_heatmap = torch.matmul(query_img_feats, img_tokens.transpose(1,2))/(self.decoder_input_dim[i]**0.5)
            pcd_keypoint_heatmap = torch.matmul(query_pcd_feats, pcd_tokens.transpose(1,2))/(self.decoder_input_dim[i]**0.5)

            if img_qk_masks != None:
                img_keypoint_heatmap = img_keypoint_heatmap.masked_fill(img_qk_masks, float("-1e5"))
            if pcd_qk_masks != None:
                pcd_keypoint_heatmap = pcd_keypoint_heatmap.masked_fill(pcd_qk_masks, float("-1e5"))

            img_keypoint_heatmap = torch.nn.functional.softmax(img_keypoint_heatmap,dim=-1)
            pcd_keypoint_heatmap = torch.nn.functional.softmax(pcd_keypoint_heatmap,dim=-1)

            img_keypoint_pixels = torch.matmul(img_keypoint_heatmap, img_pixels_list[i])
            pcd_keypoint_points = torch.matmul(pcd_keypoint_heatmap, aligned_pcd_points)

            img_keypoint_pixels_list.append(img_keypoint_pixels)
            pcd_keypoint_points_list.append(pcd_keypoint_points)

            normed_img_keypoint_pixels = self.reverse_proj(img_keypoint_pixels, intrinsics)
            transformed_pcd_keypoint_points = apply_transform(pcd_keypoint_points, RT_estimate_list[-1])

            RT4D_estimate = self.fusion_block[i](query_img_feats.detach(), query_pcd_feats.detach(), normed_img_keypoint_pixels.detach(), transformed_pcd_keypoint_points.detach())
            RT = self.RT4D2RT(RT4D_estimate)
            R = RT[:,:3,:3]
            T = RT[:,:3, 3]
            R_ = RT_estimate_list[-1][:,:3,:3]
            T_ = RT_estimate_list[-1][:,:3, 3]
            R_R = torch.matmul(R, R_)
            T_T = torch.matmul(T_.unsqueeze(1), R.transpose(-1,-2)).squeeze(1) + T
            coarse_transform = get_transform_from_rotation_translation(R_R, T_T)
            RT4D_estimate_list.append(RT4D_estimate)
            RT_estimate_list.append(coarse_transform)

        output_dict["img_keypoint_pixels_list"] = img_keypoint_pixels_list
        output_dict["pcd_keypoint_points_list"] = pcd_keypoint_points_list
        
        output_dict["RT_estimate_list"] = RT_estimate_list
        output_dict["RT4D_estimate_list"] = RT4D_estimate_list

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
