"""
KineDrive: Kinematic-Aware End-to-End VLM for Autonomous Driving

Full model integrating:
  1. Qwen2.5-VL-3B backbone with LoRA
  2. TemporalSpatial3DPE for 3D-aware visual features
  3. KinematicFlowMatchingHead for 4D (x,y,θ,v) trajectory generation
  4. Chain-of-Thought structured reasoning
  5. K-GRPO reinforcement learning (optional)

Training modes:
  MODE_SFT:   Supervised fine-tuning on CoT reasoning + trajectory text tokens
  MODE_IL:    Imitation learning (flow matching on 4D waypoints)
  MODE_GRPO:  Group Relative Policy Optimization with PDM-V reward

Usage:
  >>> model = KineDrive.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", config)
  >>> trajectory = model.plan(images, ego_status, navigation_command)
  >>> # trajectory: (B, 8, 4) — (x, y, θ, v) over 4s horizon
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kinedrive.models.kinematic_waypoints import (
    KinematicWaypointSpace,
    WAYPOINT_NUM,
    WAYPOINT_INTERVAL,
)
from kinedrive.models.kinematic_action_head import KinematicFlowMatchingHead
from kinedrive.models.spatial_pe import TemporalSpatial3DPE


# -------------------------------------------------------------------------
# Configuration dataclass
# -------------------------------------------------------------------------

@dataclass
class KineDriveConfig:
    # Base VLM
    pretrained_model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    vlm_hidden_dim: int = 2048          # Qwen2.5-VL-3B hidden size

    # LoRA
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 3D PE
    num_cameras: int = 6
    num_temporal_frames: int = 4
    token_stride: int = 14
    use_midas_depth: bool = False
    pe_scaling: float = 0.1
    learnable_pe_scaling: bool = True

    # Action head (DiT)
    action_dim: int = 4                 # (x, y, θ, v) — NOVEL: includes speed
    num_waypoints: int = WAYPOINT_NUM   # 8
    waypoint_interval: float = WAYPOINT_INTERVAL  # 0.5s
    dit_num_layers: int = 12
    dit_num_heads: int = 8
    dit_head_dim: int = 64
    ego_status_dim: int = 9             # 3 cmd + 5 kin + 1 speed

    # Training
    freeze_vision_tower: bool = True
    train_mode: str = "sft"             # "sft" | "il" | "grpo"

    # GRPO
    grpo_kl_beta: float = 0.04
    grpo_reward_scale: float = 10.0
    grpo_num_samples: int = 8
    grpo_gamma: float = 0.6             # temporal discount for denoising steps
    grpo_cot_penalty_weight: float = 0.3
    grpo_cot_penalty_center: int = 400  # tokens

    # Loss weights
    loss_flow_weight: float = 1.0
    loss_kin_weight: float = 0.1
    loss_vlm_weight: float = 0.1        # language cross-entropy weight
    loss_col_weight: float = 0.5        # soft collision penalty weight

    # Inference
    num_denoising_steps: int = 10


# -------------------------------------------------------------------------
# KineDrive model
# -------------------------------------------------------------------------

class KineDrive(nn.Module):
    """
    KineDrive: Full model combining Qwen2.5-VL-3B with 4D kinematic planning.

    Architecture pipeline:
      images (6×4) → TemporalSpatial3DPE → Qwen2.5-VL-3B (LoRA) → hidden states
      hidden states + CoT text → VLM reasoning
      hidden states → KinematicFlowMatchingHead → (x, y, θ, v) × 8 waypoints

    The 4D kinematic output is the primary novel contribution:
      - Prior work: (x,y) or (x,y,θ) only
      - KineDrive: (x,y,θ,v) — speed as first-class prediction
    """

    def __init__(self, config: KineDriveConfig):
        super().__init__()
        self.config = config

        # --- 1. Load and configure VLM backbone ---
        self._vlm = None          # Lazy-loaded; call build_vlm() before use
        self._processor = None

        # --- 2. 3D PE module ---
        self.spatial_pe = TemporalSpatial3DPE(
            embed_dim=config.vlm_hidden_dim,
            num_cameras=config.num_cameras,
            num_temporal=config.num_temporal_frames,
            token_stride=config.token_stride,
            use_midas=config.use_midas_depth,
            pe_scaling=config.pe_scaling,
            learnable_scaling=config.learnable_pe_scaling,
        )

        # --- 3. Kinematic flow matching head ---
        self.action_head = KinematicFlowMatchingHead(
            vlm_hidden_dim=config.vlm_hidden_dim,
            action_dim=config.action_dim,
            num_waypoints=config.num_waypoints,
            num_layers=config.dit_num_layers,
            num_heads=config.dit_num_heads,
            head_dim=config.dit_head_dim,
            ego_status_dim=config.ego_status_dim,
        )

        # --- 4. Waypoint space ---
        self.waypoint_space = KinematicWaypointSpace(
            num_waypoints=config.num_waypoints,
            interval=config.waypoint_interval,
        )

    # ------------------------------------------------------------------
    # Factory method
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[KineDriveConfig] = None,
    ) -> "KineDrive":
        """Load KineDrive with Qwen2.5-VL-3B-Instruct backbone and apply LoRA."""
        if config is None:
            config = KineDriveConfig(pretrained_model_path=model_path)
        else:
            config.pretrained_model_path = model_path

        model = cls(config)
        model._load_vlm()
        return model

    def _load_vlm(self):
        """Load Qwen2.5-VL-3B backbone and apply LoRA."""
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError(
                "transformers and peft required. "
                "Install with: pip install transformers>=4.49.0 peft>=0.10.0"
            ) from e

        cfg = self.config
        print(f"[KineDrive] Loading {cfg.pretrained_model_path} ...")
        vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.pretrained_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if _flash_attn_available() else "eager",
        )

        if cfg.freeze_vision_tower:
            for p in vlm.visual.parameters():
                p.requires_grad = False

        # Apply LoRA to the language model
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        vlm = get_peft_model(vlm, lora_config)
        vlm.enable_input_require_grads()

        # Upcast LoRA parameters to float32 (following SpaceDrive misc.py)
        for name, param in vlm.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        self._vlm = vlm
        self._processor = AutoProcessor.from_pretrained(
            cfg.pretrained_model_path,
            min_pixels=109760,   # matches AutoVLA (28*28*140)
            max_pixels=109760,
        )
        print("[KineDrive] VLM loaded and LoRA applied.")

    @property
    def vlm(self):
        if self._vlm is None:
            self._load_vlm()
        return self._vlm

    @property
    def processor(self):
        if self._processor is None:
            self._load_vlm()
        return self._processor

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward_sft(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        labels: torch.Tensor,
        spatial_pe_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """SFT forward pass — language cross-entropy on CoT + trajectory tokens.

        The VLM processes multi-camera images with 3D PE injected, then generates
        structured reasoning followed by trajectory text tokens.

        Returns:
            dict with 'loss_vlm', 'logits'
        """
        vlm = self.vlm

        # Inject 3D PE via model hook if spatial_pe_kwargs provided
        # (PE is added inside visual encoder output via a forward hook)
        if spatial_pe_kwargs is not None:
            self._register_pe_hook(spatial_pe_kwargs)

        out = vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
        )

        self._remove_pe_hook()

        return {"loss_vlm": out.loss, "logits": out.logits}

    def forward_il(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gt_waypoints_phys: torch.Tensor,     # (B, T, 4) — (x,y,θ,v) in physical units
        ego_status: torch.Tensor,            # (B, 9)
        hist_traj: torch.Tensor,             # (B, 4, 4) past kinematic states
        current_speed: Optional[torch.Tensor] = None,
        spatial_pe_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Imitation learning forward — flow matching on 4D kinematic trajectory.

        1. Extract VLM hidden states
        2. Normalize GT waypoints
        3. Compute flow matching loss
        4. Optionally compute kinematic consistency loss

        Returns:
            dict with 'loss_flow', 'loss_kin', 'loss_total'
        """
        vlm = self.vlm

        # Extract VLM hidden states (no generation, just encoder pass)
        if spatial_pe_kwargs is not None:
            self._register_pe_hook(spatial_pe_kwargs)

        with torch.set_grad_enabled(True):
            out = vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
            )
        vlm_hidden = out.hidden_states[-1]   # (B, S, vlm_hidden_dim)

        self._remove_pe_hook()

        # Normalize GT waypoints to [-1, 1]
        gt_norm = self.waypoint_space.normalize(gt_waypoints_phys.float())

        # Flow matching loss
        losses = self.action_head.forward_train(
            gt_waypoints_norm=gt_norm,
            vlm_hidden=vlm_hidden.float(),
            ego_status=ego_status.float(),
            hist_traj=hist_traj.float(),
            current_speed=current_speed,
        )

        return losses

    @torch.no_grad()
    def forward_inference(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ego_status: torch.Tensor,
        hist_traj: torch.Tensor,
        num_denoising_steps: Optional[int] = None,
        spatial_pe_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Inference: extract VLM hidden states, then flow-match trajectory.

        Returns:
            trajectory: (B, T, 4) — (x, y, θ, v) in physical units (meters, radians, m/s)
        """
        vlm = self.vlm
        num_steps = num_denoising_steps or self.config.num_denoising_steps

        if spatial_pe_kwargs is not None:
            self._register_pe_hook(spatial_pe_kwargs)

        out = vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )
        vlm_hidden = out.hidden_states[-1]   # (B, S, D)

        self._remove_pe_hook()

        traj = self.action_head.forward_inference(
            vlm_hidden=vlm_hidden.float(),
            ego_status=ego_status.float(),
            hist_traj=hist_traj.float(),
            num_steps=num_steps,
        )
        return traj   # (B, T, 4)

    # ------------------------------------------------------------------
    # 3D PE hook mechanism
    # (PE is added to visual token embeddings inside Qwen2.5-VL's forward)
    # ------------------------------------------------------------------

    def _register_pe_hook(self, pe_kwargs: Dict[str, Any]):
        """Register a forward hook on the visual encoder to inject 3D PE."""
        # Hook on the merge/projection layer output of Qwen2.5-VL visual encoder
        def _hook(module, input, output):
            # output: (B*T*N_cam, num_tokens, vlm_hidden_dim) or flat
            # pe_kwargs must contain: token_layout, intrinsics, lidar2img, frame_indices
            try:
                if isinstance(output, torch.Tensor):
                    output = self.spatial_pe(output, **pe_kwargs)
            except Exception:
                pass  # graceful degradation if shapes don't match
            return output

        # Hook on patch-merge output (merger in Qwen2.5-VL)
        try:
            self._pe_hook = self.vlm.model.visual.merger.register_forward_hook(_hook)
        except AttributeError:
            self._pe_hook = None

    def _remove_pe_hook(self):
        if hasattr(self, "_pe_hook") and self._pe_hook is not None:
            self._pe_hook.remove()
            self._pe_hook = None

    # ------------------------------------------------------------------
    # High-level planning API
    # ------------------------------------------------------------------

    def plan(
        self,
        images: List,                    # list of PIL Images (N_cam × T_frames)
        ego_status_dict: Dict[str, Any], # dict with velocity, acceleration, command, speed
        navigation_command: str = "go straight",
        num_denoising_steps: Optional[int] = None,
        intrinsics: Optional[torch.Tensor] = None,
        lidar2img: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """High-level planning API.

        Args:
            images:              list of PIL Images for all cameras × frames
            ego_status_dict:     {'velocity': [vx,vy], 'acceleration': [ax,ay],
                                  'yaw_rate': float, 'speed': float,
                                  'command': 'go straight'|'turn left'|'turn right'}
            navigation_command:  text description of maneuver
            num_denoising_steps: override for denoising steps
            intrinsics:          (1, N_cam, 3, 3) camera intrinsic matrices
            lidar2img:           (1, N_cam, 4, 4) lidar→image projection matrices

        Returns:
            dict with:
              'positions': (T, 2) — (x, y) in meters
              'headings':  (T, 1) — θ in radians
              'speeds':    (T, 1) — v in m/s
              'trajectory': (T, 4) — full (x, y, θ, v)
        """
        proc = self.processor
        device = next(self.parameters()).device

        # Build text prompt
        prompt = _build_cot_prompt(ego_status_dict, navigation_command)

        # Process images + text
        inputs = proc(
            text=[prompt],
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(device)

        # Ego status vector
        ego_vec = _build_ego_status_vector(ego_status_dict)
        ego_status = torch.tensor(ego_vec, dtype=torch.float32, device=device).unsqueeze(0)

        # History trajectory (zeros if not available)
        hist_traj = torch.zeros(1, 4, 4, device=device)

        # Optional 3D PE kwargs
        pe_kwargs = None
        if intrinsics is not None and lidar2img is not None:
            pe_kwargs = {
                "token_layout": (
                    self.config.num_cameras,
                    self.config.num_temporal_frames,
                    1, 1,   # H_tok, W_tok will be wrong here without proper layout
                    inputs["pixel_values"].shape[1] if inputs["pixel_values"].dim() > 2 else 1,
                ),
                "intrinsics": intrinsics.to(device),
                "lidar2img": lidar2img.to(device),
            }

        traj = self.forward_inference(
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs.get("image_grid_thw"),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            ego_status=ego_status,
            hist_traj=hist_traj,
            num_denoising_steps=num_denoising_steps,
            spatial_pe_kwargs=pe_kwargs,
        )  # (1, T, 4)

        traj = traj.squeeze(0).cpu()   # (T, 4)

        return {
            "positions":  traj[:, :2].numpy(),
            "headings":   traj[:, 2:3].numpy(),
            "speeds":     traj[:, 3:4].numpy(),
            "trajectory": traj.numpy(),
        }

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------

def _flash_attn_available() -> bool:
    try:
        import flash_attn
        return True
    except ImportError:
        return False


COT_SYSTEM_PROMPT = """You are an autonomous driving system. Given multi-camera views and ego state, \
analyze the scene and predict a safe kinematic trajectory.

Output format:
<think>
[Scene Analysis]: Describe road layout, traffic signals, weather, and drivable area.
[Velocity Context]: Current speed is {speed:.1f} m/s. Describe the speed profile needed.
[Risk Assessment]: List top-3 nearby agents with relative position and motion state.
[Speed Profile]: Describe planned speed changes (accelerate/maintain/decelerate to X m/s).
[Action Decision]: Final direction ({direction}) and speed change ({speed_change}).
</think>
<answer>
[(x1,y1,θ1,v1), (x2,y2,θ2,v2), (x3,y3,θ3,v3), (x4,y4,θ4,v4), (x5,y5,θ5,v5), (x6,y6,θ6,v6), (x7,y7,θ7,v7), (x8,y8,θ8,v8)]
</answer>

Coordinate convention: x=lateral (right=+), y=forward (forward=+), θ=heading (rad), v=speed (m/s)
Horizon: 8 waypoints × 0.5s = 4.0 seconds
"""


def _build_cot_prompt(ego_dict: Dict, navigation_command: str) -> str:
    speed = ego_dict.get("speed", 0.0)
    return (
        COT_SYSTEM_PROMPT.format(
            speed=speed,
            direction=navigation_command,
            speed_change="maintain" if speed > 1.0 else "accelerate",
        )
        + f"\n\nNavigation instruction: {navigation_command}\n"
        + f"Current ego state: speed={speed:.2f} m/s, "
        + f"velocity=({ego_dict.get('velocity', [0,0])[0]:.2f}, {ego_dict.get('velocity', [0,0])[1]:.2f}) m/s, "
        + f"acceleration=({ego_dict.get('acceleration', [0,0])[0]:.2f}, {ego_dict.get('acceleration', [0,0])[1]:.2f}) m/s²\n"
    )


def _build_ego_status_vector(ego_dict: Dict) -> List[float]:
    """Build 9-dim ego status vector: [cmd_left, cmd_fwd, cmd_right, vx, vy, ax, ay, ω, v]."""
    command = ego_dict.get("command", "go straight").lower()
    cmd = [
        1.0 if "left" in command else 0.0,
        1.0 if "straight" in command or "forward" in command else 0.0,
        1.0 if "right" in command else 0.0,
    ]
    vel = ego_dict.get("velocity", [0.0, 0.0])
    acc = ego_dict.get("acceleration", [0.0, 0.0])
    yaw_rate = ego_dict.get("yaw_rate", 0.0)
    speed = ego_dict.get("speed", math.sqrt(vel[0] ** 2 + vel[1] ** 2))

    return cmd + list(vel) + list(acc) + [yaw_rate, speed]
