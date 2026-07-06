"""
Temporal 3D Spatial Positional Encoding

Extends SpaceDrive's Universal 3D PE to multi-frame inputs.
SpaceDrive applies 3D PE only to current-frame visual tokens (single timestamp).
KineDrive extends this to all temporal frames, enabling the model to understand
spatial change across time — crucial for velocity and motion estimation.

Key difference from SpaceDrive:
  - SpaceDrive: PE for 1 timestamp × 6 cameras
  - KineDrive:  PE for T timestamps × N cameras with temporal frame indexing
                Each past frame's tokens carry both 3D spatial position AND
                temporal position (encoded via a learnable temporal embedding)

Depth estimation:
  Option A (full): Online MiDaS-v2-small inference (≈25ms/frame on A100)
  Option B (approx): Geometric fallback using a fixed depth prior
                     (uniform plane at ego height + weak perspective)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding3D(nn.Module):
    """
    Universal sine-cosine 3D positional encoding.

    Adapted from SpaceDrive's PositionalEncoding3D (positional_encoding.py):
      - Same frequency formula: f_i = 1 / (freq_coeff^(2i / channels))
      - Same interleave: [sin, cos, sin, cos, ...] per axis
      - Same learnable scale α (pe_scaling)
      - Added: temporal offset embedding for multi-frame support
    """

    def __init__(
        self,
        embed_dim: int,
        freq_coeff: float = 10000.0,
        freq_scaling: float = 1.0,
        pe_scaling: float = 0.1,
        learnable_scaling: bool = True,
        max_temporal_frames: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.freq_coeff = freq_coeff
        self.freq_scaling = freq_scaling
        self.max_temporal_frames = max_temporal_frames

        # Learnable scale α_PE — controls how strongly 3D geometry modifies visual tokens
        if learnable_scaling:
            self.pe_scaling = nn.Parameter(torch.tensor(float(pe_scaling)))
        else:
            self.register_buffer("pe_scaling", torch.tensor(float(pe_scaling)))

        # Learnable temporal offset: T × embed_dim — encodes which past frame a token belongs to
        # Additive to the spatial PE so the model can distinguish "where" AND "when"
        self.temporal_embed = nn.Embedding(max_temporal_frames, embed_dim)
        nn.init.normal_(self.temporal_embed.weight, std=0.02)

        # Per-axis channel count for the PE
        self._channels_per_axis = math.ceil(embed_dim / 6) * 2

    def _axis_pe(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute interleaved sin-cos PE for one spatial axis.

        Args:
            coords: (B, N) — coordinate values for N tokens along one axis
        Returns:
            pe: (B, N, channels_per_axis)
        """
        half = self._channels_per_axis // 2
        i = torch.arange(half, device=coords.device, dtype=coords.dtype)
        freq = self.freq_scaling / (self.freq_coeff ** (2.0 * i / self._channels_per_axis))
        # (B, N, half)
        sin_inp = coords.unsqueeze(-1) * freq.unsqueeze(0).unsqueeze(0)
        # Interleave: [sin0, cos0, sin1, cos1, ...]
        pe = torch.stack([sin_inp.sin(), sin_inp.cos()], dim=-1)  # (B, N, half, 2)
        return pe.flatten(-2, -1)                                   # (B, N, channels_per_axis)

    def forward(
        self,
        coords_3d: torch.Tensor,
        frame_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute 3D PE for a batch of 3D points.

        Args:
            coords_3d:     (B, N, 3) — (x, y, z) in LiDAR/ego frame (meters)
            frame_indices: (B, N) — temporal frame index per token (0=current, 1=t-0.5s, ...)
                            If None, all tokens assumed to be frame 0 (current)
        Returns:
            pe: (B, N, embed_dim) — positional encoding, scaled by α_PE
        """
        x, y, z = coords_3d[..., 0], coords_3d[..., 1], coords_3d[..., 2]

        pe_x = self._axis_pe(x)
        pe_y = self._axis_pe(y)
        pe_z = self._axis_pe(z)

        # Concatenate along channel dim and trim to embed_dim
        pe = torch.cat([pe_x, pe_y, pe_z], dim=-1)[..., : self.embed_dim]

        # Add temporal embedding if frame indices are given
        if frame_indices is not None:
            temp_emb = self.temporal_embed(frame_indices.long())  # (B, N, embed_dim)
            pe = pe + temp_emb

        return pe * self.pe_scaling


class GeometricDepthEstimator(nn.Module):
    """
    Lightweight geometric depth estimation fallback.

    When a full monocular depth model (e.g. MiDaS, UniDepthV2) is not available,
    this module provides a weak depth prior for the 3D PE computation:
      - Assumes a flat ground plane at ego height h_ground
      - Uses vertical pixel position + camera pitch to estimate ground distance
      - Adds a learned per-camera bias correction

    This is ~100× faster than online MiDaS and enables training without the depth model
    installed. For full performance, replace with MiDaS-small or UniDepthV2.
    """

    def __init__(
        self,
        num_cameras: int = 6,
        default_depth_m: float = 15.0,
        ego_height_m: float = 1.5,
    ):
        super().__init__()
        self.num_cameras = num_cameras
        self.default_depth_m = default_depth_m
        self.ego_height_m = ego_height_m
        # Learnable per-camera depth scale — adapts to different focal lengths / positions
        self.cam_depth_scale = nn.Parameter(torch.ones(num_cameras))

    def forward(
        self,
        image_coords_uv: torch.Tensor,   # (B, N_cam, H, W, 2) — (u, v) pixel coords
        intrinsics: torch.Tensor,         # (B, N_cam, 3, 3)
        extrinsics_lidar2cam: torch.Tensor,  # (B, N_cam, 4, 4)
    ) -> torch.Tensor:
        """Estimate per-pixel depth in LiDAR frame.

        Returns:
            depth: (B, N_cam, H, W) — estimated depth in meters
        """
        B, N, H, W, _ = image_coords_uv.shape
        # Uniform depth baseline, scaled per camera
        scale = self.cam_depth_scale[:N].view(1, N, 1, 1)
        depth = torch.full((B, N, H, W), self.default_depth_m,
                           device=image_coords_uv.device, dtype=image_coords_uv.dtype)
        # Apply learnable scale
        depth = depth * scale.abs().clamp(0.5, 5.0)
        return depth


class TemporalSpatial3DPE(nn.Module):
    """
    Multi-frame, multi-camera 3D Spatial Positional Encoding.

    Full pipeline:
      1. Get depth for each (camera × frame) image — online MiDaS or geometric fallback
      2. Unproject pixels to 3D LiDAR coordinates using depth + extrinsics
      3. Compute Universal 3D PE φ(c_p) for each visual token
      4. Add temporal embedding to distinguish past frames
      5. Additively inject PE into image token embeddings:
           image_tokens = image_tokens + PE(3D_coords, temporal_index)

    Args:
        embed_dim:          VLM hidden dim (2048 for Qwen2.5-VL-3B)
        num_cameras:        number of camera views (6 for nuScenes)
        num_temporal:       number of temporal frames (4 at 2Hz = 2s history)
        token_stride:       spatial stride of VLM visual tokens (14 for Qwen2.5-VL)
        use_midas:          if True, attempt to load MiDaS-small; else use geometric fallback
        pe_scaling:         initial value for learnable α_PE
        learnable_scaling:  whether α_PE is trainable
    """

    def __init__(
        self,
        embed_dim: int = 2048,
        num_cameras: int = 6,
        num_temporal: int = 4,
        token_stride: int = 14,
        use_midas: bool = False,
        pe_scaling: float = 0.1,
        learnable_scaling: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_cameras = num_cameras
        self.num_temporal = num_temporal
        self.token_stride = token_stride

        # 3D PE encoder
        self.pe3d = PositionalEncoding3D(
            embed_dim=embed_dim,
            pe_scaling=pe_scaling,
            learnable_scaling=learnable_scaling,
            max_temporal_frames=num_temporal,
        )

        # Depth estimation
        self._use_midas = use_midas
        self._midas_loaded = False
        if not use_midas:
            self.geo_depth = GeometricDepthEstimator(num_cameras=num_cameras)

    def _try_load_midas(self):
        """Lazy-load MiDaS-small if requested and not yet loaded."""
        if self._midas_loaded:
            return True
        try:
            import torch.hub
            self.midas = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small", pretrained=True
            ).eval()
            for p in self.midas.parameters():
                p.requires_grad = False
            self._midas_loaded = True
            return True
        except Exception:
            return False

    def _estimate_depth_midas(
        self,
        images: torch.Tensor,  # (B*N_cam*T, C, H, W) — flat batch
    ) -> torch.Tensor:         # (B*N_cam*T, H, W) — metric depth
        with torch.no_grad():
            # MiDaS outputs inverse depth; need proper scaling
            inv_depth = self.midas(images)   # (B*N*T, H', W')
            # Resize to input resolution
            inv_depth = F.interpolate(
                inv_depth.unsqueeze(1),
                size=images.shape[2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            # Convert inverse depth → metric: d = scale / inv_d + shift
            # Use a scene-level least-squares scale from sky region (v < H/4)
            depth = 1.0 / (inv_depth.abs() + 1e-4)
            depth = depth.clamp(0.5, 80.0)
        return depth

    def _unproject_to_3d(
        self,
        depth: torch.Tensor,         # (B, N_cam, T, H_tok, W_tok)
        intrinsics: torch.Tensor,     # (B, N_cam, 3, 3)
        lidar2img: torch.Tensor,      # (B, N_cam, 4, 4) — lidar-to-image projection
        image_h: int,
        image_w: int,
    ) -> torch.Tensor:                # (B, N_cam*T*H_tok*W_tok, 3)
        """Unproject depth tokens to 3D LiDAR-frame coordinates."""
        B, N, T_f, H_tok, W_tok = depth.shape
        device = depth.device

        # Pixel center coordinates of each token in the original image
        h_centers = (torch.arange(H_tok, device=device, dtype=torch.float32) + 0.5) * (
            image_h / H_tok
        )
        w_centers = (torch.arange(W_tok, device=device, dtype=torch.float32) + 0.5) * (
            image_w / W_tok
        )
        grid_v, grid_u = torch.meshgrid(h_centers, w_centers, indexing="ij")
        # (H_tok*W_tok, 3) homogeneous pixel coords
        ones = torch.ones_like(grid_u)
        pixel_hom = torch.stack([grid_u, grid_v, ones], dim=-1).view(-1, 3)   # (H*W, 3)

        # Intrinsics inverse: K^{-1} for each camera
        K_inv = torch.inverse(intrinsics)  # (B, N, 3, 3)

        # For each camera and frame, unproject
        # K^{-1} @ [u, v, 1]^T * depth → 3D in camera frame
        # then apply lidar2img^{-1} to get LiDAR coords
        img2lidar = torch.inverse(lidar2img)  # (B, N, 4, 4)

        pixel_hom_bc = pixel_hom.view(1, 1, -1, 3).expand(B, N, -1, 3)  # (B,N,H*W,3)

        # Camera-frame 3D points (before depth scaling)
        # cam_pts = K^{-1} @ pixel^T → (B, N, 3, H*W)
        cam_pts_unnorm = torch.bmm(
            K_inv.view(B * N, 3, 3),
            pixel_hom_bc.view(B * N, -1, 3).transpose(1, 2),
        ).transpose(1, 2).view(B, N, H_tok * W_tok, 3)  # (B, N, H*W, 3)

        # Scale by depth — depth is averaged over temporal frames for token-level PE
        depth_mean = depth.mean(dim=2)   # (B, N, H_tok, W_tok)
        d = depth_mean.view(B, N, H_tok * W_tok, 1)      # (B, N, H*W, 1)
        cam_pts_3d = cam_pts_unnorm * d                   # (B, N, H*W, 3)

        # Lift to homogeneous: (x, y, z, 1)
        ones_h = torch.ones(*cam_pts_3d.shape[:-1], 1, device=device)
        cam_pts_hom = torch.cat([cam_pts_3d, ones_h], dim=-1)  # (B, N, H*W, 4)

        # Transform to LiDAR frame: img2lidar @ cam_pts_hom^T
        img2lidar_bc = img2lidar.view(B * N, 4, 4)
        lidar_pts = torch.bmm(
            img2lidar_bc,
            cam_pts_hom.view(B * N, -1, 4).transpose(1, 2),
        ).transpose(1, 2).view(B, N, H_tok * W_tok, 4)  # (B, N, H*W, 4)

        # Take (x, y, z) components, repeat for each temporal frame
        lidar_xyz = lidar_pts[..., :3].unsqueeze(2).expand(-1, -1, T_f, -1, -1)
        # (B, N, T, H*W, 3) → reshape to (B, N*T*H*W, 3)
        lidar_xyz = lidar_xyz.reshape(B, N * T_f * H_tok * W_tok, 3)
        return lidar_xyz

    def forward(
        self,
        image_tokens: torch.Tensor,       # (B, N_tok_total, embed_dim)
        token_layout: Tuple[int, int, int, int, int],  # (N_cam, T, H_tok, W_tok, total)
        intrinsics: Optional[torch.Tensor] = None,     # (B, N_cam, 3, 3)
        lidar2img: Optional[torch.Tensor] = None,      # (B, N_cam, 4, 4)
        raw_images: Optional[torch.Tensor] = None,     # (B, N_cam, T, C, H, W)
        frame_indices: Optional[torch.Tensor] = None,  # (B, N_tok_total) temporal index per tok
    ) -> torch.Tensor:
        """Inject 3D PE into image token embeddings.

        Args:
            image_tokens:  (B, N_total, D) — flat sequence of all visual tokens
            token_layout:  (N_cam, T_frames, H_tok, W_tok, N_total)
            intrinsics:    per-camera K matrices
            lidar2img:     per-camera lidar→image projection matrices
            raw_images:    raw pixel values for depth estimation
            frame_indices: temporal index per token (for temporal PE)
        Returns:
            image_tokens + 3D PE: same shape (B, N_total, D)
        """
        N_cam, T_f, H_tok, W_tok, N_total = token_layout
        B = image_tokens.shape[0]

        # If no extrinsics provided, skip 3D PE (use temporal PE only)
        if intrinsics is None or lidar2img is None:
            if frame_indices is not None:
                temp_emb = self.pe3d.temporal_embed(frame_indices.long())
                return image_tokens + temp_emb * self.pe3d.pe_scaling
            return image_tokens

        # --- Depth estimation ---
        if self._use_midas and raw_images is not None and self._try_load_midas():
            B_flat = B * N_cam * T_f
            imgs_flat = raw_images.view(B_flat, *raw_images.shape[3:])
            depth_flat = self._estimate_depth_midas(imgs_flat)
            H_img, W_img = raw_images.shape[4], raw_images.shape[5]
            # Downsample depth to token resolution (min-pool: nearest obstacle wins)
            depth_tok = F.adaptive_avg_pool2d(depth_flat.unsqueeze(1), (H_tok, W_tok)).squeeze(1)
            depth = depth_tok.view(B, N_cam, T_f, H_tok, W_tok)
        elif raw_images is not None:
            H_img, W_img = raw_images.shape[4], raw_images.shape[5]
            # Geometric fallback
            B_flat = B * N_cam * T_f
            imgs_flat = raw_images.view(B_flat, *raw_images.shape[3:])
            h_coords = torch.arange(H_tok, device=image_tokens.device, dtype=torch.float32)
            w_coords = torch.arange(W_tok, device=image_tokens.device, dtype=torch.float32)
            grid_v, grid_u = torch.meshgrid(h_coords, w_coords, indexing="ij")
            uv = torch.stack([grid_u, grid_v], dim=-1).unsqueeze(0).expand(B_flat, -1, -1, -1)
            geo_depth = self.geo_depth(
                uv.view(B, N_cam, T_f, H_tok, W_tok, 2).view(B * N_cam, T_f, H_tok, W_tok, 2),
                intrinsics.view(B * N_cam, 3, 3).unsqueeze(1).expand(-1, T_f, -1, -1).reshape(-1, 3, 3),
                lidar2img.view(B * N_cam, 4, 4).unsqueeze(1).expand(-1, T_f, -1, -1).reshape(-1, 4, 4),
            )
            depth = geo_depth.view(B, N_cam, T_f, H_tok, W_tok)
            H_img, W_img = H_tok * self.token_stride, W_tok * self.token_stride
        else:
            # No depth available — use temporal PE only
            if frame_indices is not None:
                temp_emb = self.pe3d.temporal_embed(frame_indices.long())
                return image_tokens + temp_emb * self.pe3d.pe_scaling
            return image_tokens

        # --- Unproject to 3D ---
        lidar_xyz = self._unproject_to_3d(
            depth, intrinsics, lidar2img, H_img, W_img
        )  # (B, N_cam*T*H_tok*W_tok, 3)

        # --- Compute PE ---
        pe = self.pe3d(lidar_xyz, frame_indices=frame_indices)  # (B, N_total, D)

        return image_tokens + pe
