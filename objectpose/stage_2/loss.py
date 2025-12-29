import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from vision3d.ops import apply_transform,  render
from vision3d.utils.opencv import registration_with_pnp_ransac
import cv2
import os
import numpy as np

class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.mseloss = nn.MSELoss()
        self.img_radius = cfg.model.transformer.img_radius_thresold

    def diversity_loss(self, keypoints, margin):
        B, N, _ = keypoints.shape
        # Compute pairwise distances: [B, N, N]
        diff = keypoints.unsqueeze(2) - keypoints.unsqueeze(1)  # [B, N, N, 2]
        dist = torch.norm(diff, dim=-1)  # [B, N, N]
        # Penalize distances smaller than margin
        loss_mat = F.relu(margin - dist)
        # Mask out self-distances (i=j)
        mask = torch.eye(N).bool().cuda().unsqueeze(0)  # [1, N, N]
        loss_mat = loss_mat.masked_fill(mask, 0.0)
        # Normalize by number of pairs per batch item
        loss = loss_mat.sum() / (B * N * (N - 1))
        return loss
    
    def forward(self, data_dict, output_dict):
        loss_dict = {}
        intrinsics = data_dict["intrinsics"].detach()  # (B, 3, 3)
        transform = torch.stack(data_dict["transform"], dim = 0).detach()
        img_keypoint_pixels_list = output_dict["img_keypoint_pixels_list"]
        pcd_keypoint_points_list = output_dict["pcd_keypoint_points_list"]

        RT4D_estimate_list = output_dict["RT4D_estimate_list"]
        RT_estimate_list = output_dict["RT_estimate_list"]
        coarse_transform = RT_estimate_list[0]
        R6D_gt = torch.stack((transform[:,2,0], transform[:,0,0]), dim = 1)
        R6D_estimate = torch.stack((coarse_transform[:,2,0], coarse_transform[:,0,0]), dim = 1)

        T_gt = transform[:, :3, 3]
        T_estimate = coarse_transform[:, :3, 3]

        cos_theta = torch.sum(R6D_gt * R6D_estimate, dim=-1)  # (B,)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # 限制范围
        theta = torch.acos(cos_theta)  # (B,)
        # 计算 RRE (Root Rotation Error)
        RRE = torch.rad2deg(theta).detach()  # (B,)
        # 计算 RTE (Root Translation Error)
        RTE = torch.norm(T_estimate - T_gt, dim=-1).detach()  # (B,)
        loss_dict["RRE"] = RRE.mean()
        loss_dict["RTE"] = RTE.mean()

        loss = 0
        for i in range(4):
            coarse_transform = RT_estimate_list[i]
            R6D_estimate = RT4D_estimate_list[i][:,:2]
            T_estimate = RT4D_estimate_list[i][:,2:]
            R_gt = torch.matmul(transform[:,:3,:3], coarse_transform[:,:3,:3].transpose(-1,-2))
            R6D_gt = torch.stack((R_gt[:,2,0], R_gt[:,0,0]), dim = 1)

            T_trans = transform[:,:3,3]
            T_gt = T_trans - torch.matmul(coarse_transform[:,:3,3].unsqueeze(1), R_gt.transpose(-1, -2)).squeeze(1)
            T_gt = torch.stack([T_gt[:,0], T_gt[:,2]], dim = 1)

            R_loss = torch.norm(R6D_estimate - R6D_gt, dim=-1).mean()
            t_loss = torch.norm(T_estimate - T_gt, dim=-1).mean()

            cos_theta = torch.sum(R6D_gt * R6D_estimate, dim=-1)  # (B,)
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # 限制范围
            theta = torch.acos(cos_theta)  # (B,)

            # 计算 RRE (Root Rotation Error)
            RRE = torch.rad2deg(theta).detach()  # (B,)
            # 计算 RTE (Root Translation Error)
            RTE = torch.norm(T_estimate - T_gt, dim=-1).detach()  # (B,)

            loss += R_loss + t_loss
            loss_dict[f"R_loss{i}"] = R_loss
            loss_dict[f"t_loss{i}"] = t_loss
            loss_dict[f"RRE_{i}"] = RRE.mean()
            loss_dict[f"RTE_{i}"] = RTE.mean()
                
        for i in range(len(img_keypoint_pixels_list)):
            img_keypixels = img_keypoint_pixels_list[i]
            pcd_keypoints = pcd_keypoint_points_list[i]
            pcd_keypixels, _ = render(pcd_keypoints, intrinsics, extrinsics= transform, rounding=False)

            keypixel_dist = torch.norm(pcd_keypixels - img_keypixels, dim = -1)
            if i != 0:
                if query_mask.sum() == 0:
                    K2D_loss = 0
                else: 
                    mask_in = (keypixel_dist < 1e3)
                    if (query_mask & mask_in).sum() == 0:
                        K2D_loss = 0
                    else:
                        K2D_loss = keypixel_dist[query_mask & mask_in].mean() * 0.1
                        loss += K2D_loss
            else:
                mask_in = (keypixel_dist < 1e3)
                if mask_in.sum() == 0:
                    K2D_loss = 0
                else:
                    K2D_loss = keypixel_dist[mask_in].mean() * 0.1
                    loss += K2D_loss

            loss_dict[f"K2D{i}"] = K2D_loss   
            K2D_dif_loss = self.diversity_loss(img_keypixels, 16) 
            K3D_dif_loss = self.diversity_loss(pcd_keypixels, 16) 
            loss_dict[f"K2D_dif{i}"] = K2D_dif_loss
            loss_dict[f"K3D_dif{i}"] = K3D_dif_loss
            loss += K2D_dif_loss + K3D_dif_loss

            query_mask = (keypixel_dist < self.img_radius[i])
            valid_rate = query_mask.sum() / (query_mask.shape[0] * query_mask.shape[1])
            
            loss_dict[f"valid_rate{i}"] = valid_rate

        loss_dict["loss"] = loss

        return loss_dict


class EvalFunction(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def R6D_to_R(self, R6D):
        a1 = R6D[:3]
        a2 = R6D[3:]
        b1 = F.normalize(a1, p = 2, dim = 0)
        b2 = a2 - torch.dot(b1, a2)*b1
        b2 = F.normalize(b2, p = 2, dim = 0)
        b3 = torch.cross(b1, b2)
        R = torch.stack((b1,b2,b3), dim = 1)
        return R

    @torch.no_grad()
    def forward(self, data_dict, output_dict):
        intrinsics = data_dict["intrinsics"].detach()  # (B, 3, 3)
        transform = torch.stack(data_dict["transform"], dim = 0).detach()

        img_keypoint_pixels_list = output_dict["img_keypoint_pixels_list"]
        pcd_keypoint_points_list = output_dict["pcd_keypoint_points_list"]
        
        this_folder = os.getcwd()
        for j in range(4):
            pnp_store_dir = this_folder + f"/workspace/RTE_RRE_pnp_scale_{j}.txt"
            f = open(pnp_store_dir,"a") 
            pcd_keypoint_points = pcd_keypoint_points_list[j]
            img_keypoint_pixels = img_keypoint_pixels_list[j]
            for i in range(pcd_keypoint_points.shape[0]):
                estimated_transform = registration_with_pnp_ransac(
                    pcd_keypoint_points[i].cpu().numpy(),
                    img_keypoint_pixels[i].cpu().numpy(),
                    intrinsics[i].cpu().numpy(),
                    num_iterations=50000,
                    distance_tolerance=8.0,
                )

                R_i = torch.stack((transform[i,2,0], transform[i,0,0]), dim = 0).cpu().numpy()
                T_i = transform[i, :3,3].cpu().numpy()
                R_estimate = np.array([estimated_transform[2, 0], estimated_transform[0, 0]])
                T_estimate = estimated_transform[:3,3]

                cos_theta = np.dot(R_i, R_estimate)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                theta = np.arccos(cos_theta)

                RTE = np.linalg.norm(T_i - T_estimate)
                RRE = np.degrees(theta)
                
                f.write(str(RTE) + 'm ')
                f.write(str(RRE) + '°\n')
            f.close()

        RT4D_estimate_list = output_dict["RT4D_estimate_list"]
        RT_estimate_list = output_dict["RT_estimate_list"]

        this_folder = os.getcwd()
        for i in range(4):
            reg_store_dir = this_folder + f"/workspace/RTE_RRE_reg_scale_{i}.txt"
            f = open(reg_store_dir,"a") 
            coarse_transform = RT_estimate_list[i]
            R_gt = torch.matmul(transform[:,:3,:3], coarse_transform[:,:3,:3].transpose(-1,-2))

            T_trans = transform[:,:3,3]
            T_gt = T_trans - torch.matmul(coarse_transform[:,:3,3].unsqueeze(1), R_gt.transpose(-1, -2)).squeeze(1)

            R_gt = torch.stack((R_gt[:,2,0], R_gt[:,0,0]), dim = 1).cpu().numpy()
            T_gt = torch.stack([T_gt[:,0], T_gt[:,2]], dim = 1).cpu().numpy()

            R6D_estimate = F.normalize(RT4D_estimate_list[i][:,:2], dim = -1).cpu().numpy()
            T_estimate = RT4D_estimate_list[i][:,2:].cpu().numpy()

            for j in range(coarse_transform.shape[0]):
                cos_theta = np.dot(R_gt[j], R6D_estimate[j])
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                theta = np.arccos(cos_theta)

                RTE = np.linalg.norm(T_gt[j] - T_estimate[j])
                RRE = np.degrees(theta)

                f.write(str(RTE) + 'm ')
                f.write(str(RRE) + '°\n')
            f.close()

        c_precision = 0
        f_precision = 0
        return {"PIR": c_precision, "IR": f_precision}


