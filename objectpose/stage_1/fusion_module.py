import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class OverlapEstimator(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.img_mlp = nn.Sequential(
            nn.Linear(512, 512), 
            nn.LayerNorm(512),
            nn.ReLU(),
        )

        self.pcd_mlp = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.LayerNorm(512),
            nn.ReLU(),
        )
        
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 32), 
            nn.ReLU(),
            nn.Linear(32, 64), 
            nn.ReLU(),
            nn.Linear(64, 128), 
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        self.overlap_estimator = nn.Sequential(
            nn.Linear(1152, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        pcd_feats: Tensor,
        global_img_feats: Tensor,
        pcd_points_c: Tensor,
        pcd_lengths: Tensor,
    ) -> Tensor:
        global_img_feats_c = self.img_mlp(global_img_feats)
        global_img_feats_c = global_img_feats_c.repeat_interleave(pcd_lengths, dim=0)
        pcd_feats_c = self.pcd_mlp(pcd_feats)
        pcd_points_feats = self.point_mlp(pcd_points_c)

        fusion_feats = torch.cat([pcd_feats_c, global_img_feats_c, pcd_points_feats], dim = 1)
        overlap_mask = self.overlap_estimator(fusion_feats).squeeze()
        return overlap_mask

class VertexPred(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.mlp1 = nn.Sequential(
            nn.Linear(1, 32), 
            nn.ReLU(),
            nn.Linear(32, 64), 
            nn.ReLU(),
            nn.Linear(64, 128), 
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(3, 32), 
            nn.ReLU(),
            nn.Linear(32, 64), 
            nn.ReLU(),
            nn.Linear(64, 128), 
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        self.token_mlp = nn.Sequential(
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Linear(256, 128), 
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        
        self.feat_mlp = nn.Sequential(
            nn.Linear(1024, 256), 
            nn.ReLU(),
            nn.Linear(256, 128), 
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        self.points_mlp1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.points_mlp2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        self.center_estimator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        self.dir_estimator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(
        self,
        pcd_masks: Tensor,
        pcd_tokens: Tensor,
        pcd_feats: Tensor,
        pcd_points: Tensor,
        pcd_lengths: Tensor,
    ) -> Tensor:
        '''
        pcd_masks_pred:            [2, 40960]
        pcd_points:                [1, 40960, 3]
        '''
        points_embed = self.mlp2(pcd_points)
        masks_embed = self.mlp1(pcd_masks.unsqueeze(-1))

        pcd_tokens_c = self.token_mlp(pcd_tokens)
        pcd_tokens_c = pcd_tokens_c.repeat_interleave(pcd_lengths, dim=0)
        pcd_feats_c = self.feat_mlp(pcd_feats)

        points_feats = torch.cat([points_embed, masks_embed, pcd_tokens_c, pcd_feats_c], dim = -1)
        points_feats = self.points_mlp1(points_feats)

        global_feats_list = []
        B = len(pcd_lengths)
        start_idx = 0
        for i in range(B):
            lengths = pcd_lengths[i]
            end_idx = start_idx + lengths
            
            points_feats_c = points_feats[start_idx:end_idx]
            points_global = torch.mean(points_feats_c, dim = 0, keepdim = True)

            points_feats_c = torch.cat([points_feats_c, points_global.expand_as(points_feats_c)], dim = -1)
            points_feats_c = self.points_mlp2(points_feats_c).mean(dim=0)

            global_feats_list.append(points_feats_c)
            start_idx = end_idx
            
        global_feats = torch.stack(global_feats_list, dim = 0)
        center = self.center_estimator(global_feats)
        dir = F.normalize(self.dir_estimator(global_feats), dim = -1)

        vertex = torch.cat([center, dir], dim = 1)
        return vertex