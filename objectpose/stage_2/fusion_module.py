from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from vision3d.layers import TransformerLayer

class CrossModalFusionModule(nn.Module):
    def __init__(
        self,
        img_input_dim: int,
        pcd_input_dim: int,
        query_input_dim: int,
        query_output_dim: int,
        num_heads: int = 4,
        activation_fn: str = "ReLU",
        dropout: Optional[float] = None,
        use_embedding: bool = True,
    ):
        super().__init__()
        self.use_embedding = use_embedding
        self.img_input_dim = img_input_dim
        self.pcd_input_dim = pcd_input_dim

        self.img_in_proj = nn.Sequential(
            nn.Linear(img_input_dim, query_input_dim), 
            # We accidentally omitted the following 2-layer network during training. You can include it to match the setup used for KITTI.
            # nn.ReLU(),
            # nn.Linear(query_input_dim, query_input_dim), 
            nn.LayerNorm(query_input_dim),
        )

        self.pcd_in_proj = nn.Sequential(
            nn.Linear(pcd_input_dim, query_input_dim), 
            # We accidentally omitted the following 2-layer network during training. You can include it to match the setup used for KITTI.
            # nn.ReLU(),
            # nn.Linear(query_input_dim, query_input_dim), 
            nn.LayerNorm(query_input_dim),
        )
        
        self.query_out_proj = nn.Sequential(
            nn.Linear(query_input_dim, query_output_dim), 
            # We accidentally omitted the following 2-layer network during training. You can include it to match the setup used for KITTI.
            # nn.ReLU(),
            # nn.Linear(query_output_dim, query_output_dim), 
            nn.LayerNorm(query_output_dim),
        )

        self.self_attention = nn.ModuleList()
        self.self_attention.append(TransformerLayer(query_input_dim, num_heads, dropout=dropout, act_cfg=activation_fn))
        self.self_attention.append(TransformerLayer(query_input_dim, num_heads, dropout=dropout, act_cfg=activation_fn))

        self.cross_attention = nn.ModuleList()
        self.cross_attention.append(TransformerLayer(query_input_dim, num_heads, dropout=dropout, act_cfg=activation_fn))
        self.cross_attention.append(TransformerLayer(query_input_dim, num_heads, dropout=dropout, act_cfg=activation_fn))


    def forward(
        self,
        query_feats: Tensor,
        img_feats: Tensor,
        pcd_feats: Tensor,
        img_masks: Optional[Tensor] = None,
        pcd_masks: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        img_tokens = self.img_in_proj(img_feats)
        pcd_tokens = self.pcd_in_proj(pcd_feats)

        query_list = []
        query_feats_s1 = self.self_attention[0](query_feats, query_feats, query_feats)
        query_feats_c1 = self.cross_attention[0](query_feats_s1, img_tokens, img_tokens, qk_masks = img_masks)
        query_list.append(query_feats_c1) 

        query_feats_s2 = self.self_attention[1](query_feats_c1, query_feats_c1, query_feats_c1)
        query_feats_c2 = self.cross_attention[1](query_feats_s2, pcd_tokens, pcd_tokens, qk_masks = pcd_masks)
        query_list.append(query_feats_c2) 

        query_output = self.query_out_proj(query_feats_c2)       
        query_list.append(query_output) 

        return query_list, img_tokens, pcd_tokens

class FeatureFusion(nn.Module):
    def __init__(
        self,
        query_input_dim: int,
    ):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(3, 32), 
            nn.ReLU(),
            nn.Linear(32, 64), 
            nn.ReLU(),
            nn.Linear(64, 128), 
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(2, 32), 
            nn.ReLU(),
            nn.Linear(32, 64), 
            nn.ReLU(),
            nn.Linear(64, 128), 
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        self.pose_mlp1 = nn.Sequential(
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Linear(256, 256), 
            nn.ReLU(),
        )

        self.pose_mlp2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        self.rotation_estimator = nn.Sequential(
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

        self.translation_estimator = nn.Sequential(
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
        
        if query_input_dim == 512:
            self.img_query_proj = nn.Sequential(
                nn.Linear(512, 256), 
                nn.ReLU(),
                nn.Linear(256, 128), 
                nn.LayerNorm(128),
                nn.ReLU(),
            )

            self.pcd_query_proj = nn.Sequential(
                nn.Linear(512, 256), 
                nn.ReLU(),
                nn.Linear(256, 128), 
                nn.LayerNorm(128),
                nn.ReLU(),
            )
        elif query_input_dim == 256:
            self.img_query_proj = nn.Sequential(
                nn.Linear(256, 128), 
                nn.ReLU(),
                nn.Linear(128, 128), 
                nn.LayerNorm(128),
                nn.ReLU(),
            )

            self.pcd_query_proj = nn.Sequential(
                nn.Linear(256, 128), 
                nn.ReLU(),
                nn.Linear(128, 128), 
                nn.LayerNorm(128),
                nn.ReLU(),
            )
        elif query_input_dim == 128:
            self.img_query_proj = nn.Sequential(
                nn.Linear(128, 128), 
                nn.ReLU(),
                nn.Linear(128, 128), 
                nn.LayerNorm(128),
                nn.ReLU(),
            )

            self.pcd_query_proj = nn.Sequential(
                nn.Linear(128, 128), 
                nn.ReLU(),
                nn.Linear(128, 128), 
                nn.LayerNorm(128),
                nn.ReLU(),
            )

    def forward(
        self,
        img_query: Tensor,
        pcd_query: Tensor,
        img_keypoints: Tensor,
        pcd_keypoints: Tensor,
    ) -> Tensor:
        '''
        query_pcd_feats:            [1, num_queries, 128]
        query_img_feats:            [1, num_queries, 128]
        img_keypoints:              [1, num_queries, 2]    (normalized)
        pcd_keypoints:              [1, num_queries, 3]
        '''
        pixel_embed = self.mlp2(img_keypoints)
        points_embed = self.mlp1(pcd_keypoints)

        img_query_feats = self.img_query_proj(img_query)
        pcd_query_feats = self.pcd_query_proj(pcd_query)

        pose_feats = torch.cat([pixel_embed, img_query_feats, points_embed, pcd_query_feats], dim = -1)

        pose_feats = self.pose_mlp1(pose_feats)
        pose_global = torch.mean(pose_feats, dim = 1, keepdim = True)

        pose_feats = torch.cat([pose_feats, pose_global.expand_as(pose_feats)], dim = -1)
        pose_feats = self.pose_mlp2(pose_feats).mean(dim=1)

        r = F.normalize(self.rotation_estimator(pose_feats), dim = -1)

        t = self.translation_estimator(pose_feats)

        RT_estimate = torch.cat([r, t], dim = 1)

        return RT_estimate

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
        with torch.no_grad():
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
        with torch.no_grad():
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
