import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import numpy as np

class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.mseloss = nn.MSELoss()
        self.BCELoss = nn.BCEWithLogitsLoss()
        
    def forward(self, data_dict, output_dict):
        """
        data_dict:
            transform: [B, 4, 4]
            gt_mask: [B, 1, H, W]
            pred_vertex: [B, 2]
            mask_confidence: [B, 1, H, W]
        output_dict:
            pred_vertex: [B, 2]
        """
        loss_dict = {}
        transform = torch.stack(data_dict["transform"], dim = 0).detach()

        mask_confidence = output_dict["mask_confidence"]
        gt_mask = output_dict["gt_mask"].detach()
        pred_vertex = output_dict["pred_vertex"]

        center = torch.matmul(- transform[:,:3,3].unsqueeze(1), transform[:, :3, :3]).squeeze(1)
        x_center = center[:,0]
        z_center = center[:,2]
        xz_center = torch.stack([x_center, z_center], dim = 1)
        dir = torch.stack([transform[:,2,0], transform[:,0,0]], dim = 1)

        cls_loss = self.BCELoss(mask_confidence, gt_mask.float())
        t_loss_mse = self.mseloss(pred_vertex[:,:2], xz_center) * 0.1
        R_loss_mse = self.mseloss(pred_vertex[:,2:], dir)

        loss = cls_loss + R_loss_mse + t_loss_mse
        loss_dict["cls_loss"] = cls_loss
        loss_dict["R_loss_mse"] = R_loss_mse
        loss_dict["t_loss_mse"] = t_loss_mse
        loss_dict["loss"] = loss
        return loss_dict

class EvalFunction(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    
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
        
    @torch.no_grad()
    def forward(self, data_dict, output_dict):
        transform = torch.stack(data_dict["transform"], dim = 0).detach()
        pred_vertex = output_dict["pred_vertex"]

        RT = self.vertex2RT(pred_vertex)
        
        R6D_estimate = torch.stack((RT[:,2,0], RT[:,0,0]), dim = 1).cpu().numpy()
        R6D_gt = torch.stack((transform[:,2,0], transform[:,0,0]), dim = 1).cpu().numpy()

        T_gt = transform[:,:3,3].cpu().numpy()
        T_estimate = RT[:,:3,3].cpu().numpy()

        this_folder = os.getcwd()
        reg_store_dir = this_folder + "/workspace/RTE_RRE_record.txt"
        f = open(reg_store_dir,"a")

        for j in range(pred_vertex.shape[0]):
            cos_theta = np.dot(R6D_gt[j], R6D_estimate[j])
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)

            RTE = np.linalg.norm(T_gt[j] - T_estimate[j])
            RRE = np.degrees(theta)
            f.write(str(RTE) + 'm ')
            f.write(str(RRE) + '°\n')
        f.close()


        gt_mask = output_dict["gt_mask"].detach()
        after_mask = output_dict["pred_mask"].detach()
        mask_confidence = output_dict["mask_confidence"].detach()

        before_mask = (mask_confidence > 0.0)
        
        this_folder = os.getcwd()
        record_before_dir = this_folder + "/workspace/without_GPDM_overlap_detection_record.txt"
        record_after_dir = this_folder + "/workspace/with_GPDM_overlap_detection_record.txt"

        f1 = open(record_before_dir, "a") 
        f2 = open(record_after_dir, "a") 

        TP_TN_before = (gt_mask == before_mask).sum(-1).cpu().numpy()
        TP_TN_after = (gt_mask == after_mask).sum(-1).cpu().numpy()

        TP_before = (gt_mask & before_mask).sum(-1).cpu().numpy()
        TP_after = (gt_mask & after_mask).sum(-1).cpu().numpy()

        T_before = gt_mask.sum(-1).cpu().numpy()
        T_after = gt_mask.sum(-1).cpu().numpy()

        P_before = before_mask.sum(-1).cpu().numpy()
        P_after = after_mask.sum(-1).cpu().numpy()
        
        f1.write(str(TP_TN_before)+ ' ' + str(TP_before) + ' ' + str(T_before) + ' ' + str(P_before) + '\n')
        f2.write(str(TP_TN_after)+ ' ' + str(TP_after) + ' ' + str(T_after) + ' ' + str(P_after) + '\n')
        f1.close()
        f2.close()

        c_precision = 0
        f_precision = 0
        return {"PIR": c_precision, "IR": f_precision}



