"""
4D Kinematic Waypoint Space

Novel contribution: first E2E VLM that predicts (x, y, θ, v) — speed as a first-class output.

All prior work (AutoVLA, UniDriveVLA, ReCogDrive, MindDriver, SpaceDrive) encodes speed
implicitly through positional displacement between consecutive waypoints. This module makes
speed explicit, enabling downstream controllers to track a precise velocity profile.

Coordinate convention (ego-centric, lidar frame):
  x: lateral   (right = +x)
  y: forward   (forward = +y)
  θ: heading   (counter-clockwise from +y, radians)
  v: speed     (m/s, always ≥ 0)

Waypoints:
  8 timesteps at 0.5s intervals → 4-second planning horizon
  Normalization: each dimension mapped to [-1, 1] using nuScenes-derived statistics

Speed derivation from CAN bus:
  v_t = sqrt(vx_t^2 + vy_t^2)  from ego velocity in world frame
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# nuScenes-derived normalization statistics for (x, y, θ, v).
# x/y from SpaceDrive+/MindDriver analysis; θ from ReCogDrive; v from CAN bus.
KINEMATIC_STATS = {
    # (mean, std) for z-score normalization in flow matching latent space.
    # We then map to [-1, 1] via tanh for DDIM denoising compatibility.
    "x":   {"min": -10.0,  "max": 10.0},   # lateral: ±10m covers lane changes
    "y":   {"min":  -1.0,  "max": 60.0},   # forward: 0–60m at 60km/h over 4s
    "theta": {"min": -math.pi / 2, "max": math.pi / 2},  # ±90° heading change
    "v":   {"min":  0.0,   "max": 30.0},   # speed: 0–108 km/h (urban+highway)
}

# Horizon configuration
WAYPOINT_NUM = 8            # number of future waypoints
WAYPOINT_INTERVAL = 0.5     # seconds between waypoints
WAYPOINT_HORIZON = WAYPOINT_NUM * WAYPOINT_INTERVAL   # 4.0 seconds

# Physical constants for kinematic constraint checking
MAX_ACCEL    = 5.0    # m/s^2: max comfortable longitudinal acceleration (nuScenes stats)
MAX_DECEL    = 6.0    # m/s^2: max emergency deceleration
MAX_JERK     = 4.13   # m/s^3: comfort jerk threshold (from AutoVLA PDMScorer)
MAX_YAW_RATE = 0.95   # rad/s: comfort yaw rate


@dataclass
class KinematicWaypoints:
    """Container for 4D kinematic waypoints."""
    positions:  torch.Tensor   # (B, T, 2)  — (x, y) in meters
    headings:   torch.Tensor   # (B, T, 1)  — θ in radians
    speeds:     torch.Tensor   # (B, T, 1)  — v in m/s

    @property
    def batch_size(self) -> int:
        return self.positions.shape[0]

    @property
    def num_steps(self) -> int:
        return self.positions.shape[1]

    def as_tensor(self) -> torch.Tensor:
        """Return (B, T, 4) — (x, y, θ, v)."""
        return torch.cat([self.positions, self.headings, self.speeds], dim=-1)

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "KinematicWaypoints":
        """Build from (B, T, 4) tensor."""
        assert t.shape[-1] == 4, f"Expected 4D waypoints, got {t.shape[-1]}D"
        return cls(
            positions=t[..., :2],
            headings=t[..., 2:3],
            speeds=t[..., 3:4],
        )


class KinematicWaypointSpace(nn.Module):
    """
    Encodes/decodes 4D kinematic waypoints for flow matching.

    Maps real-world (x, y, θ, v) to normalized [-1, 1]^4 space and back.
    Provides kinematic consistency checking and loss computation.
    """

    def __init__(
        self,
        num_waypoints: int = WAYPOINT_NUM,
        interval: float = WAYPOINT_INTERVAL,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.interval = interval
        self.eps = eps

        # Register normalization bounds as buffers for device-portable use.
        stats = KINEMATIC_STATS
        lo = torch.tensor([stats["x"]["min"],   stats["y"]["min"],
                           stats["theta"]["min"], stats["v"]["min"]])
        hi = torch.tensor([stats["x"]["max"],   stats["y"]["max"],
                           stats["theta"]["max"], stats["v"]["max"]])
        self.register_buffer("norm_lo", lo)
        self.register_buffer("norm_hi", hi)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize(self, waypoints: torch.Tensor) -> torch.Tensor:
        """Map (x, y, θ, v) ∈ physical domain to [-1, 1]^4.

        Args:
            waypoints: (B, T, 4)
        Returns:
            normalized: (B, T, 4), each dim ∈ [-1, 1]
        """
        lo = self.norm_lo.to(waypoints)
        hi = self.norm_hi.to(waypoints)
        return 2.0 * (waypoints - lo) / (hi - lo + self.eps) - 1.0

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        """Inverse of normalize — map [-1, 1]^4 back to physical units.

        Args:
            normalized: (B, T, 4)
        Returns:
            waypoints: (B, T, 4), physical units
        """
        lo = self.norm_lo.to(normalized)
        hi = self.norm_hi.to(normalized)
        return (normalized + 1.0) / 2.0 * (hi - lo + self.eps) + lo

    # ------------------------------------------------------------------
    # Speed supervision from CAN bus
    # ------------------------------------------------------------------

    @staticmethod
    def speeds_from_can_bus(
        future_ego_velocities: torch.Tensor,
    ) -> torch.Tensor:
        """Derive per-waypoint speed from ego velocity vectors.

        Args:
            future_ego_velocities: (B, T, 2) — (vx, vy) in m/s at each future step
        Returns:
            speeds: (B, T, 1) — scalar speed = ||[vx, vy]||_2
        """
        return future_ego_velocities.norm(dim=-1, keepdim=True)

    @staticmethod
    def speed_from_position_diff(
        positions: torch.Tensor,
        interval: float = WAYPOINT_INTERVAL,
    ) -> torch.Tensor:
        """Estimate speed from finite differences of positions.

        Args:
            positions: (B, T, 2) — (x, y) positions at each timestep
            interval:  time between steps in seconds
        Returns:
            speeds: (B, T, 1) — estimated speed; first step uses forward diff
        """
        # Forward diff for t=0, central diff for t=1..T-2, backward diff for t=T-1
        diffs = torch.zeros_like(positions)
        diffs[:, 0]  = positions[:, 1] - positions[:, 0]
        diffs[:, 1:-1] = (positions[:, 2:] - positions[:, :-2]) / 2.0
        diffs[:, -1]   = positions[:, -1] - positions[:, -2]
        return (diffs / interval).norm(dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    # Kinematic Consistency Loss (novel)
    # ------------------------------------------------------------------

    def kinematic_consistency_loss(
        self,
        waypoints: torch.Tensor,
        current_speed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Penalize physically implausible speed transitions and negative speeds.

        Computes three soft penalties:
          1. Excessive acceleration: |v_{t+1} - v_t| / Δt > a_max
          2. Negative speed: v_t < 0 (vehicle cannot reverse)
          3. Initial speed continuity: if current_speed given, |v_0 - v_curr| must be small

        Args:
            waypoints:      (B, T, 4) in physical units — (x, y, θ, v)
            current_speed:  (B,) current ego speed at t=0, optional
        Returns:
            loss: scalar
        """
        v = waypoints[..., 3]          # (B, T)
        dt = self.interval

        # --- 1. Acceleration constraint ---
        dv = v[:, 1:] - v[:, :-1]     # (B, T-1)
        accel = dv / dt
        # Soft ReLU: penalize excess above max comfortable decel/accel
        accel_pen = torch.relu(accel.abs() - MAX_ACCEL).pow(2).mean()

        # --- 2. Non-negative speed ---
        neg_speed_pen = torch.relu(-v).pow(2).mean()

        # --- 3. Initial speed continuity ---
        if current_speed is not None:
            v0_pen = (v[:, 0] - current_speed.to(v)).pow(2).mean()
        else:
            v0_pen = torch.tensor(0.0, device=waypoints.device)

        return accel_pen + neg_speed_pen + v0_pen

    # ------------------------------------------------------------------
    # Heading derivation
    # ------------------------------------------------------------------

    @staticmethod
    def heading_from_positions(positions: torch.Tensor) -> torch.Tensor:
        """Compute heading from consecutive position vectors.

        Args:
            positions: (B, T, 2) — (x, y)
        Returns:
            headings: (B, T, 1) — atan2(dy, dx) in radians
        """
        diffs = torch.zeros_like(positions)
        diffs[:, :-1] = positions[:, 1:] - positions[:, :-1]
        diffs[:, -1]  = diffs[:, -2]   # repeat last diff
        # atan2(dy, dx): dy = forward (+y), dx = lateral (+x)
        # heading = angle from +y axis = atan2(dx, dy) for ego convention
        return torch.atan2(diffs[..., 0:1], diffs[..., 1:2] + 1e-6)

    # ------------------------------------------------------------------
    # Build supervision tensor from nuScenes data
    # ------------------------------------------------------------------

    def build_supervision(
        self,
        gt_positions: torch.Tensor,
        gt_velocities: Optional[torch.Tensor] = None,
        gt_headings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Assemble 4D supervision tensor for training.

        Args:
            gt_positions:  (B, T, 2) ground-truth (x, y) in ego-frame
            gt_velocities: (B, T, 2) ground-truth (vx, vy), optional
            gt_headings:   (B, T, 1) ground-truth heading, optional
        Returns:
            supervision: (B, T, 4) — (x, y, θ, v) in physical units
        """
        # Heading: use GT if available, else derive from positions
        if gt_headings is not None:
            headings = gt_headings
        else:
            headings = self.heading_from_positions(gt_positions)

        # Speed: use GT velocities if available, else derive from positions
        if gt_velocities is not None:
            speeds = self.speeds_from_can_bus(gt_velocities)
        else:
            speeds = self.speed_from_position_diff(gt_positions, self.interval)

        return torch.cat([gt_positions, headings, speeds], dim=-1)

    def forward(self, waypoints: torch.Tensor) -> torch.Tensor:
        """Normalize waypoints for flow matching training."""
        return self.normalize(waypoints)
