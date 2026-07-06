"""
KineDrive: Kinematic-Aware End-to-End VLM for Autonomous Driving

Novel contributions:
  1. 4D Kinematic Waypoints (x, y, θ, v) — speed as first-class output
  2. Kinematic Flow Matching (KFM) — 4D trajectory generation with dynamics constraints
  3. Temporal 3D Spatial Context Injection — 3D PE over multi-frame multi-camera inputs
  4. Kinematic GRPO (K-GRPO) — RL with speed-profile smoothness reward

Base model: Qwen2.5-VL-3B-Instruct
"""

from kinedrive.models.kinedrive import KineDrive
from kinedrive.models.kinematic_action_head import KinematicFlowMatchingHead
from kinedrive.models.spatial_pe import TemporalSpatial3DPE
from kinedrive.models.kinematic_waypoints import KinematicWaypointSpace

__all__ = [
    "KineDrive",
    "KinematicFlowMatchingHead",
    "TemporalSpatial3DPE",
    "KinematicWaypointSpace",
]
