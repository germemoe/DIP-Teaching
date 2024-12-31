from os import system
from turtle import width
from scipy import linalg
import sklearn.naive_bayes
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass
import numpy as np
import cv2


class GaussianRenderer(nn.Module):
    def __init__(self, image_height: int, image_width: int):
        super().__init__()
        self.H = image_height
        self.W = image_width
        
        # Pre-compute pixel coordinates grid
        y, x = torch.meshgrid(
            torch.arange(image_height, dtype=torch.float32),
            torch.arange(image_width, dtype=torch.float32),
            indexing='ij'
        )
        # Shape: (H, W, 2)
        self.register_buffer('pixels', torch.stack([x, y], dim=-1))


    def compute_projection(
        self,
        means3D: torch.Tensor,          # (N, 3)
        covs3d: torch.Tensor,           # (N, 3, 3)
        K: torch.Tensor,                # (3, 3)
        R: torch.Tensor,                # (3, 3)
        t: torch.Tensor                 # (3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = means3D.shape[0]
        
        # 1. Transform points to camera space
        cam_points = means3D @ R.T + t.unsqueeze(0) # (N, 3)
        
        # 2. Get depths before projection for proper sorting and clipping
        depths = cam_points[:, 2].clamp(min=1.)  # (N, )
        
        # 3. Project to screen space using camera intrinsics
        screen_points = cam_points @ K.T  # (N, 3)

        if (screen_points[..., 2:3] == 0).any():
            print("jb")

        means2D = screen_points[..., :2] / screen_points[..., 2:3] # (N, 2)
        
        # 4. Transform covariance to camera space and then to 2D
        # Compute Jacobian of perspective projection
        J_proj = torch.zeros((N, 2, 3), device=means3D.device)
        ### FILL:
        ### J_proj = ...

        tx, ty, tz = torch.unbind(cam_points, dim=-1)
        tz2 = tz**2
        width = self.W
        height = self.H
        fx = K[..., 0, 0, None]
        fy = K[..., 1, 1, None]
        cx = K[..., 0, 2, None]
        cy = K[..., 1, 2, None]
        tan_fovx = 0.5 * width / fx
        tan_fovy = 0.5 * height / fy

        lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
        lim_x_neg = cx / fx + 0.3 * tan_fovx
        lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
        lim_y_neg = cy / fy + 0.3 * tan_fovy
        tx = tz * torch.clamp(tx / tz, min=-lim_x_neg, max=lim_x_pos)
        ty = tz * torch.clamp(ty / tz, min=-lim_y_neg, max=lim_y_pos)

        O = torch.zeros(N, device=cam_points.device, dtype=cam_points.dtype)
        J_proj = torch.stack(
            [fx / tz, O, -fx * tx / tz2, O, fy / tz, -fy * ty / tz2], dim=-1
        ).reshape(N, 2, 3)
        # Transform covariance to camera space
        ### FILL: Aplly world to camera rotation to the 3d covariance matrix
        ### covs_cam = ...  # (N, 3, 3)
        covs_cam = R @ covs3d @ R.T
        # Project to 2D
        covs2D = torch.bmm(J_proj, torch.bmm(covs_cam, J_proj.permute(0, 2, 1)))  # (N, 2, 2)

        if torch.isnan(means2D).any():
            print("jb")
        if torch.isnan(covs2D).any():
            print("jb")
        if torch.isnan(depths).any():
            print("jb")

        return means2D, covs2D, depths

    def compute_gaussian_values(
        self,
        means2D: torch.Tensor,    # (N, 2)
        covs2D: torch.Tensor,     # (N, 2, 2)
        pixels: torch.Tensor      # (H, W, 2)
    ) -> torch.Tensor:           # (N, H, W)
        N = means2D.shape[0]
        H, W = pixels.shape[:2]
        
        # Compute offset from mean (N, H, W, 2)
        dx = pixels.unsqueeze(0) - means2D.reshape(N, 1, 1, 2)
        
        # Add small epsilon to diagonal for numerical stability
        eps = 1e-4
        covs2D = covs2D + eps * torch.eye(2, device=covs2D.device).unsqueeze(0)
        
        # Compute determinant for normalization
        ### FILL: compute the gaussian values
        ### gaussian = ... ## (N, H, W)
        det = (
            covs2D[..., 0, 0] * covs2D[..., 1, 1]
            - covs2D[..., 0, 1] * covs2D[..., 1, 0]
        )
        det = det.clamp(min=1e-4)

        covs2D_inv = torch.zeros_like(covs2D)
        covs2D_inv[..., 0, 0] = covs2D[..., 1, 1] / det
        covs2D_inv[..., 1, 1] = covs2D[..., 0, 0] / det
        covs2D_inv[..., 1, 0] = -(covs2D[..., 0, 1] + covs2D[..., 1, 0]) / 2.0 / det
        covs2D_inv[..., 0, 1] = -(covs2D[..., 0, 1] + covs2D[..., 1, 0]) / 2.0 / det

        # P = -0.5 * torch.einsum('npki,nij,npkj->npk', dx, torch.linalg.inv(covs2D), dx)
        P = (-0.5 * dx.unsqueeze(-2) @ covs2D_inv.unsqueeze(1).unsqueeze(1) @ dx.unsqueeze(-1)).clamp(max = 80)
        # c = (2*torch.pi*torch.sqrt(torch.linalg.det(covs2D))).unsqueeze(1).unsqueeze(2).repeat(1, H, W)
        # gaussian = torch.exp(P)/c
        gaussian = torch.exp(P).squeeze(-1).squeeze(-1)
        if torch.isinf(gaussian).any():
            raise NotImplementedError('00000000000000nan')
        return gaussian

    def forward(
            self,
            means3D: torch.Tensor,          # (N, 3)
            covs3d: torch.Tensor,           # (N, 3, 3)
            colors: torch.Tensor,           # (N, 3)
            opacities: torch.Tensor,        # (N, 1)
            K: torch.Tensor,                # (3, 3)
            R: torch.Tensor,                # (3, 3)
            t: torch.Tensor                 # (3, 1)
    ) -> torch.Tensor:
        N = means3D.shape[0]
        
        # 1. Project to 2D, means2D: (N, 2), covs2D: (N, 2, 2), depths: (N,)
        means2D, covs2D, depths = self.compute_projection(means3D, covs3d, K, R, t)
        
        # 2. Depth mask
        valid_mask = (depths > 1.) & (depths < 50.0)  # (N,)
        
        # 3. Sort by depth
        indices = torch.argsort(depths, dim=0, descending=False)  # (N, )
        means2D = means2D[indices]      # (N, 2)
        covs2D = covs2D[indices]       # (N, 2, 2)
        colors = colors[ indices]       # (N, 3)
        opacities = opacities[indices] # (N, 1)
        valid_mask = valid_mask[indices] # (N,)
        
        # 4. Compute gaussian values
        gaussian_values = self.compute_gaussian_values(means2D, covs2D, self.pixels)  # (N, H, W)
        
        # 5. Apply valid mask
        gaussian_values = gaussian_values * valid_mask.view(N, 1, 1)  # (N, H, W)

        # 6. Alpha composition setup
        alphas = torch.clamp_max(opacities.view(N, 1, 1) * gaussian_values, 0.999)  # (N, H, W)
        colors = colors.view(N, 3, 1, 1).expand(-1, -1, self.H, self.W)  # (N, 3, H, W)
        colors = colors.permute(0, 2, 3, 1)  # (N, H, W, 3)
        
        # 7. Compute weights
        ### FILL:
        ### weights = ... # (N, H, W)
        T = torch.cumprod(1 - alphas, dim = 0)
        weights = alphas * T
        # 8. Final rendering
        rendered = (weights.unsqueeze(-1) * colors).sum(dim=0)  # (H, W, 3)
        return rendered


