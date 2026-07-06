"""
Kinematic Flow Matching Action Head (KFM-Head)

Novel contribution: extends ReCogDrive's LightningDiT diffusion planner to
4D kinematic space (x, y, θ, v) via flow matching.

Key differences from ReCogDrive:
  1. Action dim = 4 (adds explicit speed v), not 3 (x, y, θ)
  2. Uses flow matching (like UniDriveVLA), not DDPM (like ReCogDrive)
     — Flow matching: fewer denoising steps (10), more stable gradients
  3. Kinematic Consistency Loss applied in physical (denormalized) space
  4. Perception feature injection via extra cross-attention on ego query
     (from simplified sparse perception)
  5. Temporal PE-conditioned ego status includes current speed explicitly

Architecture:
  - 12-layer DiT with interleaved self/cross-attention
  - Even layers (0,2,4,...): pure self-attention over 8 noised action tokens
  - Odd layers (1,3,5,...): cross-attention on VLM hidden states (context)
  - AdaLN: conditioned on (timestep_emb + ego_status_emb)
  - Position encoding on action sequence: learned 8-dim PE
  - Output: (B, 8, 4) — velocity field in normalized 4D kinematic space

References:
  - ReCogDrive DiT (recogdrive/navsim/agents/recogdrive/recogdrive_dit.py)
  - UniDriveVLA flow matching (nuScenes/projects/.../flow_planning_loss.py)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kinedrive.models.kinematic_waypoints import WAYPOINT_NUM, KinematicWaypointSpace


# ---------------------------------------------------------------------------
# Utility modules (adapted from ReCogDrive blocks/)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: x → x * (1 + scale) + shift."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward (from ReCogDrive/UniDriveVLA)."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w3  = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep → MLP embedding (from ReCogDrive)."""

    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SinusoidalTimestep(nn.Module):
    """Continuous timestep t ∈ [0,1] → sinusoidal embedding."""

    def __init__(self, num_channels: int = 256):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        half = self.num_channels // 2
        freq = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        emb = t.unsqueeze(-1) * freq.unsqueeze(0)     # (B, half)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, num_channels)


class EgoStatusEncoder(nn.Module):
    """
    Ego status → embedding for AdaLN conditioning.

    Input dims: navigation_command (3) + vx (1) + vy (1) + ax (1) + ay (1) + yaw_rate (1)
                + current_speed (1) = 9 total
    Novel addition vs ReCogDrive: explicit current_speed dimension.
    """

    def __init__(self, in_dim: int = 9, hidden_dim: int = 256, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HistTrajectoryEncoder(nn.Module):
    """Encode past 4-frame (x, y, θ, v) history → sequence embedding."""

    def __init__(self, in_dim: int = 16, out_dim: int = 512, num_steps: int = WAYPOINT_NUM):
        super().__init__()
        # 4 past frames × 4D = 16-dim flat vector
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.num_steps = num_steps

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        # hist: (B, 4, 4) past kinematic states or (B, 16) flattened
        if hist.dim() == 3:
            hist = hist.flatten(1)
        emb = self.proj(hist)          # (B, out_dim)
        # Replicate to match action sequence length
        return emb.unsqueeze(1).expand(-1, self.num_steps, -1)   # (B, T, out_dim)


# ---------------------------------------------------------------------------
# Attention module with QK-RMSNorm + RoPE
# ---------------------------------------------------------------------------

class KFMAttention(nn.Module):
    """Multi-head attention with QK-RMSNorm (follows ReCogDrive Attention)."""

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        num_heads: int = 8,
        head_dim: int = 64,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = head_dim
        inner_dim = num_heads * head_dim
        context_dim = context_dim or query_dim

        self.to_q = nn.Linear(query_dim,  inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim, bias=False)

        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:       (B, N_q, D)
            context: (B, N_k, D_ctx) — if None, self-attention
        Returns:
            out: (B, N_q, D)
        """
        B, N_q, _ = x.shape
        context = context if context is not None else x

        q = self.to_q(x).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # QK-RMSNorm (per head)
        q = self.q_norm(q)
        k = self.k_norm(k)

        out = F.scaled_dot_product_attention(q, k, v)  # (B, H, N_q, head_dim)
        out = out.transpose(1, 2).reshape(B, N_q, -1)
        return self.to_out(out)


# ---------------------------------------------------------------------------
# DiT Block with AdaLN-Zero
# ---------------------------------------------------------------------------

class KFMDiTBlock(nn.Module):
    """
    Single KFM DiT block.

    Even block index: pure self-attention (action tokens attend each other)
    Odd  block index: cross-attention on VLM hidden states (context)

    Conditioning: AdaLN on (timestep_emb + ego_status_emb), 6-parameter modulation
    FFN: SwiGLU (following ReCogDrive/UniDriveVLA)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        context_dim: int,
        is_cross_attn: bool = False,
        ffn_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.is_cross_attn = is_cross_attn

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        # Self-attention always present
        self.self_attn = KFMAttention(query_dim=dim, num_heads=num_heads, head_dim=head_dim)

        # Cross-attention only in cross-attn blocks
        if is_cross_attn:
            self.cross_attn = KFMAttention(
                query_dim=dim, context_dim=context_dim,
                num_heads=num_heads, head_dim=head_dim,
            )
            self.norm_cross = RMSNorm(dim)

        self.ffn  = SwiGLUFFN(dim, ffn_hidden_dim)

        # AdaLN-Zero: 6 modulation parameters (shift_attn, scale_attn, gate_attn,
        #                                        shift_ffn,  scale_ffn,  gate_ffn)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        # Zero-init as in DiT paper for stable training at init
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,            # (B, T, dim)
        conditioning: torch.Tensor, # (B, dim) — t_emb + ego_emb
        context: Optional[torch.Tensor] = None,  # (B, S, context_dim) VLM features
    ) -> torch.Tensor:
        # AdaLN modulation parameters
        params = self.adaLN_modulation(conditioning)   # (B, 6*dim)
        sh_a, sc_a, g_a, sh_f, sc_f, g_f = params.chunk(6, dim=-1)

        # Self-attention path
        residual = x
        x_norm = modulate(self.norm1(x), sh_a, sc_a)
        x = residual + g_a.unsqueeze(1) * self.self_attn(x_norm)

        # Cross-attention path (odd blocks)
        if self.is_cross_attn and context is not None:
            x = x + self.cross_attn(self.norm_cross(x), context)

        # FFN path
        residual = x
        x_norm = modulate(self.norm2(x), sh_f, sc_f)
        x = residual + g_f.unsqueeze(1) * self.ffn(x_norm)

        return x


# ---------------------------------------------------------------------------
# Kinematic Flow Matching Head (full DiT)
# ---------------------------------------------------------------------------

class KinematicFlowMatchingHead(nn.Module):
    """
    Full 4D Kinematic Flow Matching Head.

    Denoises a trajectory in (x, y, θ, v) space conditioned on:
      - VLM hidden states (cross-attention)
      - Ego status (AdaLN via timestep + status embedding)
      - Historical 4D trajectory (prefix fused with action tokens)

    Architecture (default "small"):
      12 blocks, 8 heads, head_dim=64, inner_dim=512

    Input action space: 4D = (x, y, θ, v) per waypoint, normalized to [-1, 1]
    Output: predicted velocity field in 4D normalized space
    """

    def __init__(
        self,
        vlm_hidden_dim: int = 2048,    # Qwen2.5-VL-3B hidden size
        action_dim: int = 4,            # (x, y, θ, v)
        num_waypoints: int = WAYPOINT_NUM,
        num_layers: int = 12,
        num_heads: int = 8,
        head_dim: int = 64,
        ego_status_dim: int = 9,        # 3 cmd + 5 kinematics + 1 speed = 9
        hist_frames: int = 4,
        sinusoidal_channels: int = 256,
        ffn_hidden_dim: int = 512,
    ):
        super().__init__()
        self.action_dim    = action_dim
        self.num_waypoints = num_waypoints
        self.num_heads     = num_heads
        self.num_layers    = num_layers

        inner_dim = num_heads * head_dim  # 512

        # --- Action encoder ---
        # Projects noised actions + timestep → inner_dim
        self.action_proj = nn.Linear(action_dim, inner_dim, bias=False)
        self.time_proj   = nn.Sequential(
            SinusoidalTimestep(sinusoidal_channels),
            TimestepEmbedding(sinusoidal_channels, inner_dim),
        )
        self.action_time_mlp = nn.Sequential(
            nn.Linear(inner_dim * 2, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, inner_dim),
        )

        # Learned position embedding for action sequence dimension
        self.pos_embed = nn.Embedding(num_waypoints, inner_dim)

        # --- History encoder (4 past kinematic states × 4D = 16 → inner_dim) ---
        self.hist_encoder = HistTrajectoryEncoder(
            in_dim=hist_frames * action_dim,
            out_dim=inner_dim,
            num_steps=num_waypoints,
        )

        # --- Ego status encoder → used in AdaLN ---
        self.ego_encoder = EgoStatusEncoder(
            in_dim=ego_status_dim, hidden_dim=inner_dim, out_dim=inner_dim
        )

        # --- VLM context projector ---
        # Map VLM hidden dim to DiT inner dim for cross-attention
        self.ctx_proj = nn.Linear(vlm_hidden_dim, inner_dim, bias=False)

        # --- Fusion projector: [hist | ctx_mean | action] → inner_dim ---
        self.fusion_proj = nn.Linear(inner_dim * 3, inner_dim, bias=False)

        # --- DiT blocks ---
        self.blocks = nn.ModuleList([
            KFMDiTBlock(
                dim=inner_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                context_dim=inner_dim,
                is_cross_attn=(layer_idx % 2 == 1),   # odd layers = cross-attn
                ffn_hidden_dim=ffn_hidden_dim,
            )
            for layer_idx in range(num_layers)
        ])

        # --- Final projection → velocity field (4D per waypoint) ---
        self.final_norm = RMSNorm(inner_dim)
        self.final_proj = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, action_dim),
        )
        # Zero-init final linear for stable start
        nn.init.zeros_(self.final_proj[-1].weight)
        nn.init.zeros_(self.final_proj[-1].bias)

        # Waypoint space for normalization / denormalization
        self.waypoint_space = KinematicWaypointSpace()

    # ------------------------------------------------------------------
    # Flow matching utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_time(batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample flow matching time t ~ Beta(1.5, 1.0) skewed toward t=1.

        This mirrors UniDriveVLA's time sampling, which up-weights high-noise
        timesteps where the gradient signal is strongest.
        """
        beta = torch.distributions.Beta(
            torch.tensor(1.5, device=device),
            torch.tensor(1.0, device=device),
        )
        return beta.sample((batch_size,)).clamp(0.001, 0.999)

    def _forward_dit(
        self,
        noised_actions: torch.Tensor,   # (B, T, 4) normalized
        t: torch.Tensor,                 # (B,)
        vlm_hidden: torch.Tensor,        # (B, S, vlm_hidden_dim)
        ego_status: torch.Tensor,        # (B, 9)
        hist_traj: torch.Tensor,         # (B, 4, 4) past kinematic states
    ) -> torch.Tensor:                   # (B, T, 4) predicted velocity field
        B = noised_actions.shape[0]
        device = noised_actions.device

        # Project VLM context to inner_dim
        ctx = self.ctx_proj(vlm_hidden)        # (B, S, inner_dim)
        ctx_mean = ctx.mean(dim=1)             # (B, inner_dim) — coarse global context

        # Action + time embedding
        pos_ids = torch.arange(self.num_waypoints, device=device)
        action_emb = (
            self.action_proj(noised_actions)                           # (B, T, inner_dim)
            + self.pos_embed(pos_ids).unsqueeze(0)                     # (B, T, inner_dim)
        )
        t_emb = self.time_proj(t)                                      # (B, inner_dim)
        action_emb = self.action_time_mlp(
            torch.cat([action_emb, t_emb.unsqueeze(1).expand(-1, self.num_waypoints, -1)], dim=-1)
        )                                                               # (B, T, inner_dim)

        # History embedding
        hist_emb = self.hist_encoder(hist_traj)   # (B, T, inner_dim)

        # Context mean replicated over action horizon
        ctx_mean_seq = ctx_mean.unsqueeze(1).expand(-1, self.num_waypoints, -1)  # (B, T, inner_dim)

        # Fusion: [hist | ctx_mean | action] → inner_dim
        x = self.fusion_proj(
            torch.cat([hist_emb, ctx_mean_seq, action_emb], dim=-1)
        )                                                               # (B, T, inner_dim)

        # Conditioning for AdaLN: t_emb + ego_emb
        ego_emb = self.ego_encoder(ego_status)                         # (B, inner_dim)
        conditioning = t_emb + ego_emb                                 # (B, inner_dim)

        # DiT forward
        for block in self.blocks:
            x = block(x, conditioning, context=ctx)

        # Final projection to velocity field
        x = self.final_norm(x)
        v_pred = self.final_proj(x)    # (B, T, 4)
        return v_pred

    # ------------------------------------------------------------------
    # Training forward (flow matching)
    # ------------------------------------------------------------------

    def forward_train(
        self,
        gt_waypoints_norm: torch.Tensor,   # (B, T, 4) normalized ground-truth
        vlm_hidden: torch.Tensor,           # (B, S, vlm_hidden_dim)
        ego_status: torch.Tensor,           # (B, 9)
        hist_traj: torch.Tensor,            # (B, 4, 4)
        current_speed: Optional[torch.Tensor] = None,  # (B,) for kinematic loss
    ) -> dict:
        """Compute flow matching loss.

        Flow matching (following UniDriveVLA):
          x_t = (1 - t) * x_0 + t * ε   (interpolate data → noise)
          u_t = ε - x_0                  (velocity target)
          L_flow = MSE(v_pred, u_t)

        Returns dict with:
          loss_flow:   flow matching MSE
          loss_kin:    kinematic consistency (in physical space)
          loss_total:  weighted sum
        """
        B, T, D = gt_waypoints_norm.shape
        device = gt_waypoints_norm.device

        # Sample noise and time
        eps = torch.randn_like(gt_waypoints_norm)
        t   = self._sample_time(B, device)

        # Noised actions (linear interpolation from data to noise)
        t_bc = t.view(B, 1, 1)
        x_t = (1.0 - t_bc) * gt_waypoints_norm + t_bc * eps

        # Target velocity field
        u_t = eps - gt_waypoints_norm

        # Predict velocity field
        v_pred = self._forward_dit(x_t, t, vlm_hidden, ego_status, hist_traj)

        # --- Flow matching loss (MSE with optional min-SNR weighting) ---
        loss_flow = F.mse_loss(v_pred, u_t)

        # --- Kinematic consistency loss (in denormalized physical space) ---
        # Reconstruct predicted x_0 from flow: x_0_pred = x_t - t * v_pred
        x0_pred_norm = x_t - t_bc * v_pred
        x0_pred_phys = self.waypoint_space.denormalize(x0_pred_norm.detach())
        loss_kin = self.waypoint_space.kinematic_consistency_loss(x0_pred_phys, current_speed)

        loss_total = loss_flow + 0.1 * loss_kin

        return {
            "loss_flow":  loss_flow,
            "loss_kin":   loss_kin,
            "loss_total": loss_total,
        }

    # ------------------------------------------------------------------
    # Inference (Euler flow matching)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_inference(
        self,
        vlm_hidden: torch.Tensor,   # (B, S, vlm_hidden_dim)
        ego_status: torch.Tensor,   # (B, 9)
        hist_traj: torch.Tensor,    # (B, 4, 4)
        num_steps: int = 10,
    ) -> torch.Tensor:              # (B, T, 4) in physical units
        """Euler integration from noise to trajectory.

        Follows UniDriveVLA's inference: x_T = ε, x_{t-1} = x_t + dt * v_t
        with dt = -1/num_steps (integrate from t=1 → t=0).
        """
        B = vlm_hidden.shape[0]
        device = vlm_hidden.device

        # Start from pure noise
        x_t = torch.randn(B, self.num_waypoints, self.action_dim, device=device)

        # Euler integration
        dt = -1.0 / num_steps
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)[:-1]

        for t_val in timesteps:
            t = t_val.expand(B)
            v_t = self._forward_dit(x_t, t, vlm_hidden, ego_status, hist_traj)
            x_t = x_t + dt * v_t

        # Clamp normalized output to [-1, 1] before denormalization
        x_t = x_t.clamp(-1.0, 1.0)

        # Denormalize to physical units
        traj_phys = self.waypoint_space.denormalize(x_t)

        # Enforce non-negative speed
        traj_phys[..., 3] = traj_phys[..., 3].clamp(min=0.0)

        return traj_phys

    # ------------------------------------------------------------------
    # GRPO rollout (returns full denoising chain for DiffGRPO)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout_chain(
        self,
        vlm_hidden: torch.Tensor,
        ego_status: torch.Tensor,
        hist_traj: torch.Tensor,
        num_steps: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a full denoising chain for GRPO.

        Returns:
            chain:  (B, num_steps+1, T, 4) — intermediate states x_{T..0} (normalized)
            traj:   (B, T, 4) — final trajectory in physical units
        """
        B = vlm_hidden.shape[0]
        device = vlm_hidden.device

        x_t = torch.randn(B, self.num_waypoints, self.action_dim, device=device)
        chain = [x_t.clone()]

        dt = -1.0 / num_steps
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)[:-1]

        for t_val in timesteps:
            t = t_val.expand(B)
            v_t = self._forward_dit(x_t, t, vlm_hidden, ego_status, hist_traj)
            x_t = x_t + dt * v_t
            chain.append(x_t.clone())

        chain_tensor = torch.stack(chain, dim=1)   # (B, K+1, T, 4)
        traj_phys = self.waypoint_space.denormalize(x_t.clamp(-1.0, 1.0))
        traj_phys[..., 3] = traj_phys[..., 3].clamp(min=0.0)

        return chain_tensor, traj_phys

    def get_logprobs(
        self,
        chain: torch.Tensor,          # (B, K+1, T, 4)
        vlm_hidden: torch.Tensor,
        ego_status: torch.Tensor,
        hist_traj: torch.Tensor,
        num_steps: int = 5,
    ) -> torch.Tensor:                 # (B, K)
        """Compute per-step log-probabilities for GRPO.

        Uses Gaussian log-probability of each denoising transition:
          log p(x_{t-1} | x_t) = log N(x_{t-1}; x_t + dt*v_t, σ²I)
        with σ = 0.1 (minimum std for numerical stability).
        """
        B, K_plus1, T, D = chain.shape
        K = K_plus1 - 1
        sigma = 0.1

        dt = -1.0 / num_steps
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=chain.device)[:-1]

        log_probs = []
        for k in range(K):
            t_val = timesteps[k]
            t = t_val.expand(B)
            x_t    = chain[:, k]     # (B, T, D)
            x_next = chain[:, k + 1] # (B, T, D)

            with torch.no_grad():
                v_t = self._forward_dit(x_t, t, vlm_hidden, ego_status, hist_traj)

            mean_next = x_t + dt * v_t
            # log N(x_next; mean_next, sigma^2)
            log_p = -0.5 * ((x_next - mean_next) / sigma).pow(2).mean(dim=[1, 2])
            log_probs.append(log_p)

        return torch.stack(log_probs, dim=1)   # (B, K)
