"""
nuScenes Dataset for KineDrive Training

Loads nuScenes scenes and builds 4D kinematic supervision tensors (x, y, θ, v).

Speed supervision:
  - Primary: nuScenes CAN bus `vehicle_speed` (km/h → m/s)
  - Fallback: derived from positional differencing of gt_ego_fut_trajs

Data format:
  Each sample is a dict with:
    input_ids, attention_mask, pixel_values, image_grid_thw  — VLM inputs
    gt_waypoints_phys   (8, 4) — (x, y, θ, v) GT in physical units
    ego_status          (9,)   — [cmd(3), vx, vy, ax, ay, ω, v]
    hist_traj           (4, 4) — past 4 kinematic states
    current_speed       scalar — ego speed at t=0
    token               str    — nuScenes sample token (for GRPO reward lookup)
    hascot              bool   — whether this sample has CoT annotation
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from kinedrive.models.kinematic_waypoints import KinematicWaypointSpace


class NuScenesKinematicDataset(Dataset):
    """
    nuScenes dataset building 4D kinematic (x,y,θ,v) supervision.

    Extends the standard nuScenes planning dataset by:
      1. Adding explicit speed dimension from CAN bus or velocity differencing
      2. Building heading from trajectory differential
      3. Packaging past 4-frame kinematic history for the action head
      4. Optionally loading pre-generated CoT annotations for SFT

    Args:
        data_root:       path to processed nuScenes JSON scene files
        sensor_root:     path to nuScenes sensor blobs
        split:           'train' or 'val'
        processor:       Qwen2.5-VL processor instance
        waypoint_space:  KinematicWaypointSpace instance
        use_cot:         if True, load CoT annotations from `cot_path`
        cot_path:        path to JSON with CoT annotations (keyed by token)
        max_samples:     limit dataset size (useful for debugging)
        image_size:      (H, W) target image size
    """

    # nuScenes camera ordering used in CoT prompt
    CAMERA_ORDER = [
        "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_BACK",
    ]

    def __init__(
        self,
        data_root: str,
        sensor_root: str,
        split: str = "train",
        processor=None,
        waypoint_space: Optional[KinematicWaypointSpace] = None,
        use_cot: bool = True,
        cot_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        image_size: Tuple[int, int] = (480, 640),
    ):
        self.data_root    = Path(data_root)
        self.sensor_root  = Path(sensor_root)
        self.split        = split
        self.processor    = processor
        self.waypoint_space = waypoint_space or KinematicWaypointSpace()
        self.use_cot      = use_cot
        self.image_size   = image_size

        # Load scene info
        info_file = self.data_root / f"nuscenes_infos_{split}.pkl"
        if info_file.exists():
            with open(info_file, "rb") as f:
                infos = pickle.load(f)
            self.scene_infos = infos["infos"] if "infos" in infos else infos
        else:
            # Fall back to JSON scene files from AutoVLA format
            json_dir = self.data_root / split
            self.scene_infos = []
            if json_dir.exists():
                for p in sorted(json_dir.glob("*.json")):
                    with open(p) as f:
                        self.scene_infos.append(json.load(f))

        if max_samples:
            self.scene_infos = self.scene_infos[:max_samples]

        # Load CoT annotations if provided
        self.cot_annotations: Dict[str, str] = {}
        if use_cot and cot_path and os.path.exists(cot_path):
            with open(cot_path) as f:
                self.cot_annotations = json.load(f)

        print(f"[KineDriveDataset] Loaded {len(self.scene_infos)} samples from {split} split")

    def __len__(self) -> int:
        return len(self.scene_infos)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        info = self.scene_infos[idx]
        token = info.get("token", str(idx))

        # --- Load images ---
        images = self._load_images(info)

        # --- Build ego status vector ---
        ego_status = self._build_ego_status(info)
        current_speed = float(np.sqrt(ego_status[3] ** 2 + ego_status[4] ** 2))

        # --- Build 4D kinematic waypoints (GT) ---
        gt_waypoints_phys = self._build_gt_waypoints(info)   # (8, 4)

        # --- Build past kinematic history ---
        hist_traj = self._build_hist_traj(info)              # (4, 4)

        # --- CoT annotation ---
        cot = self.cot_annotations.get(token, "")
        hascot = bool(cot)

        # --- Build VLM text prompt ---
        from kinedrive.models.kinedrive import _build_cot_prompt, _build_ego_status_vector
        ego_dict = self._info_to_ego_dict(info)
        command = info.get("instruction", "go straight")
        prompt_text = _build_cot_prompt(ego_dict, command)

        if hascot:
            # Append ground-truth CoT + trajectory text as supervision
            gt_traj_text = self._waypoints_to_text(gt_waypoints_phys)
            full_response = f"<think>\n{cot}\n</think>\n<answer>\n{gt_traj_text}\n</answer>"
            conversation = [
                {"role": "user",      "content": prompt_text},
                {"role": "assistant", "content": full_response},
            ]
        else:
            gt_traj_text = self._waypoints_to_text(gt_waypoints_phys)
            full_response = f"<think>\nThis is a straightforward scenario.\n</think>\n<answer>\n{gt_traj_text}\n</answer>"
            conversation = [
                {"role": "user",      "content": prompt_text},
                {"role": "assistant", "content": full_response},
            ]

        # --- Tokenize for VLM ---
        if self.processor is not None:
            try:
                vlm_inputs = self._tokenize(conversation, images)
            except Exception:
                vlm_inputs = {
                    "input_ids":      torch.zeros(1, dtype=torch.long),
                    "attention_mask": torch.zeros(1, dtype=torch.long),
                    "pixel_values":   torch.zeros(1, 3, 224, 224),
                    "labels":         torch.full((1,), -100, dtype=torch.long),
                }
        else:
            vlm_inputs = {}

        return {
            **vlm_inputs,
            "gt_waypoints_phys": torch.tensor(gt_waypoints_phys, dtype=torch.float32),
            "ego_status":        torch.tensor(ego_status, dtype=torch.float32),
            "hist_traj":         torch.tensor(hist_traj, dtype=torch.float32),
            "current_speed":     torch.tensor(current_speed, dtype=torch.float32),
            "token":             token,
            "hascot":            hascot,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_images(self, info: Dict) -> List:
        """Load PIL images for all cameras."""
        from PIL import Image
        images = []
        for cam in self.CAMERA_ORDER:
            cam_info = info.get("cams", {}).get(cam, {})
            img_path = cam_info.get("data_path", "")
            if img_path and os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                img = img.resize((self.image_size[1], self.image_size[0]))
            else:
                img = Image.new("RGB", (self.image_size[1], self.image_size[0]), color=128)
            images.append(img)
        return images

    def _build_ego_status(self, info: Dict) -> np.ndarray:
        """Build 9-dim ego status: [cmd(3), vx, vy, ax, ay, ω, speed]."""
        command = info.get("gt_ego_fut_cmd", [0, 1, 0])   # default: go straight
        vel = np.array(info.get("gt_ego_his_diff", [[0, 0]] * 4)[-1])  # last diff
        acc = np.zeros(2)
        if "gt_ego_his_diff" in info and len(info["gt_ego_his_diff"]) > 1:
            diffs = np.array(info["gt_ego_his_diff"])
            if len(diffs) >= 2:
                acc = (diffs[-1] - diffs[-2]) / 0.5   # finite diff acceleration
        yaw_rate = float(info.get("yaw_rate", 0.0))
        speed = float(np.linalg.norm(vel))
        return np.concatenate([command, vel, acc, [yaw_rate, speed]]).astype(np.float32)

    def _build_gt_waypoints(self, info: Dict) -> np.ndarray:
        """Build (8, 4) = (x, y, θ, v) GT waypoints.

        Speed is derived from CAN bus if available, else from position differencing.
        Heading is derived from consecutive position differences.
        """
        # Future positions — (T, 2) in ego frame
        fut_trajs = np.array(info.get("gt_ego_fut_trajs", np.zeros((9, 2))))
        # gt_ego_fut_trajs[0] = current position (0,0), [1:9] = future 8 steps
        positions = fut_trajs[1:9, :2]   # (8, 2) — take 8 future steps

        if len(positions) < 8:
            # Pad with last known position
            pad = np.repeat(positions[-1:], 8 - len(positions), axis=0)
            positions = np.vstack([positions, pad])

        # Headings from position differences
        diffs = np.diff(np.vstack([np.zeros((1, 2)), positions]), axis=0)  # (8, 2)
        diffs[-1] = diffs[-2] if len(diffs) > 1 else diffs[-1]
        headings = np.arctan2(diffs[:, 0], diffs[:, 1] + 1e-6)  # (8,)

        # Speed from CAN bus or differencing
        can_speed = info.get("future_speeds_ms", None)
        if can_speed is not None and len(can_speed) >= 8:
            speeds = np.array(can_speed[:8])
        else:
            # Derive from position differences: ||Δpos|| / Δt
            speeds = np.linalg.norm(diffs / 0.5, axis=1)  # (8,)

        waypoints = np.stack([
            positions[:, 0],  # x (lateral)
            positions[:, 1],  # y (forward)
            headings,          # θ (heading)
            speeds,            # v (speed, m/s)
        ], axis=-1).astype(np.float32)  # (8, 4)

        return waypoints

    def _build_hist_traj(self, info: Dict) -> np.ndarray:
        """Build (4, 4) past kinematic states = (x, y, θ, v)."""
        hist_raw = np.array(info.get("gt_ego_his_trajs", np.zeros((5, 2))))
        # Typically (5, 2): current + 4 past, take past 4
        if hist_raw.shape[0] >= 5:
            hist_pos = hist_raw[-5:-1, :2]   # (4, 2)
        else:
            hist_pos = np.zeros((4, 2))

        diffs = np.diff(np.vstack([hist_pos, hist_pos[-1:]]), axis=0)
        headings = np.arctan2(diffs[:, 0], diffs[:, 1] + 1e-6)
        speeds   = np.linalg.norm(diffs / 0.5, axis=1)

        return np.stack([
            hist_pos[:, 0], hist_pos[:, 1], headings, speeds
        ], axis=-1).astype(np.float32)  # (4, 4)

    def _info_to_ego_dict(self, info: Dict) -> Dict:
        cmd = info.get("gt_ego_fut_cmd", [0, 1, 0])
        vel = info.get("gt_ego_his_diff", [[0.0, 0.0]])[-1]
        instruction = info.get("instruction", "go straight")
        command_str = "go straight"
        if cmd[0] > 0.5:
            command_str = "turn right"
        elif cmd[1] > 0.5:
            command_str = "go straight"
        elif cmd[2] > 0.5:
            command_str = "turn left"
        return {
            "velocity": vel,
            "acceleration": [0.0, 0.0],
            "yaw_rate": 0.0,
            "speed": float(np.linalg.norm(vel)),
            "command": command_str,
        }

    @staticmethod
    def _waypoints_to_text(waypoints: np.ndarray) -> str:
        """Convert (8, 4) waypoints to text: [(x1,y1,θ1,v1), ...]."""
        parts = [
            f"({wp[0]:+.2f},{wp[1]:+.2f},{wp[2]:+.3f},{wp[3]:.2f})"
            for wp in waypoints
        ]
        return "[" + ", ".join(parts) + "]"

    def _tokenize(self, conversation: List[Dict], images: List) -> Dict[str, torch.Tensor]:
        """Tokenize conversation with images for Qwen2.5-VL."""
        proc = self.processor

        # Build text with image placeholders
        text = proc.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )

        inputs = proc(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True,
        )

        # Build labels: mask user tokens, keep assistant response
        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()

        # Find assistant start token position — mask everything before
        # Qwen2.5-VL assistant token sequence: [151644, 77091]
        ASSISTANT_TOKENS = [151644, 77091]
        for i in range(len(input_ids) - len(ASSISTANT_TOKENS)):
            if input_ids[i : i + len(ASSISTANT_TOKENS)].tolist() == ASSISTANT_TOKENS:
                labels[: i + len(ASSISTANT_TOKENS)] = -100
                break
        else:
            labels[:] = -100  # Safety: mask all if not found

        inputs["labels"] = labels.unsqueeze(0)
        return {k: v.squeeze(0) if v.dim() > 1 and k != "pixel_values" else v
                for k, v in inputs.items()}


# -------------------------------------------------------------------------
# Data collator
# -------------------------------------------------------------------------

class KinematicDataCollator:
    """Collate function for KineDrive datasets."""

    IGNORE_INDEX = -100

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        # Stack tensor fields
        result: Dict[str, Any] = {}

        tensor_keys = ["gt_waypoints_phys", "ego_status", "hist_traj", "current_speed"]
        for key in tensor_keys:
            if key in batch[0]:
                result[key] = torch.stack([b[key] for b in batch])

        # VLM inputs with padding
        if "input_ids" in batch[0]:
            result["input_ids"]      = _pad_sequence([b["input_ids"]      for b in batch], 0)
            result["attention_mask"] = _pad_sequence([b["attention_mask"]  for b in batch], 0)
            result["labels"]         = _pad_sequence([b["labels"]          for b in batch], self.IGNORE_INDEX)

        if "pixel_values" in batch[0]:
            result["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)

        if "image_grid_thw" in batch[0]:
            result["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], dim=0)

        result["tokens"] = [b["token"] for b in batch]
        result["hascot"]  = [b.get("hascot", False) for b in batch]

        return result


def _pad_sequence(tensors: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    max_len = max(t.shape[-1] for t in tensors)
    result = torch.full((len(tensors), max_len), pad_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        result[i, -t.shape[-1]:] = t   # left-pad (matches AutoVLA convention)
    return result
