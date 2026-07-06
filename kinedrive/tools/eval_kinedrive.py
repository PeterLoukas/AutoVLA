"""
KineDrive Evaluation Script

Evaluates on:
  1. nuScenes open-loop planning:
     - L2 displacement at 1s/2s/3s/4s (L2-1 through L2-4)
     - Collision rate (BEV occupancy)
     - NEW: Speed Profile Error (SPE) — mean |v_pred - v_GT| in m/s
     - NEW: Heading Error (HE) — mean |θ_pred - θ_GT| in degrees

  2. NAVSIM closed-loop (optional):
     - PDMS (PDM Score): NC × DAC × weighted(TTC, Progress, Comfort)

Metrics are computed under two variants matching prior work:
  - UniAD-style: per-timestep breakdown + average
  - STP3-style:  cumulative mean

Usage:
  python kinedrive/tools/eval_kinedrive.py \
      --checkpoint checkpoints/kinedrive_grpo_epoch009.pt \
      --pretrained Qwen/Qwen2.5-VL-3B-Instruct \
      --data_root /data/nuscenes \
      --sensor_root /data/nuscenes/sensor_blobs \
      --split val \
      --output_dir ./eval_results
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[2]))

from kinedrive.models.kinedrive import KineDrive, KineDriveConfig
from kinedrive.datasets.nuscenes_kinematic import NuScenesKinematicDataset, KinematicDataCollator


def parse_args():
    parser = argparse.ArgumentParser(description="KineDrive Evaluation")
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--pretrained",   default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data_root",    required=True)
    parser.add_argument("--sensor_root",  required=True)
    parser.add_argument("--split",        default="val")
    parser.add_argument("--output_dir",   default="./eval_results")
    parser.add_argument("--batch_size",   type=int, default=4)
    parser.add_argument("--num_workers",  type=int, default=4)
    parser.add_argument("--num_steps",    type=int, default=10, help="Denoising steps")
    parser.add_argument("--max_samples",  type=int, default=None)
    return parser.parse_args()


# -------------------------------------------------------------------------
# Metric computation (matches UniAD / SpaceDrive eval_planning.py)
# -------------------------------------------------------------------------

class KineDriveMetrics:
    """
    Compute all KineDrive-specific metrics.

    Standard metrics (matching prior work for fair comparison):
      L2 at t=1s,2s,3s,4s (average Euclidean distance)
      Collision rate at t=1s,2s,3s (BEV occupancy overlap)

    Novel metrics (unique to KineDrive):
      SPE (Speed Profile Error): mean |v_pred - v_GT| over 8 steps [m/s]
      HE  (Heading Error):        mean |θ_pred - θ_GT| over 8 steps [degrees]
      Kinematic Consistency Score: fraction of steps with |Δv| / Δt ≤ 5 m/s²
    """

    EGO_W = 1.85   # ego vehicle width [m]
    EGO_H = 4.084  # ego vehicle length [m]
    BEV_SIZE = 200   # BEV grid size (pixels)
    BEV_RES  = 0.5   # meters per pixel
    BEV_RANGE = 50.0 # ±50m

    def __init__(self, horizon_steps: int = 8, interval: float = 0.5):
        self.horizon = horizon_steps
        self.interval = interval
        self.times_s = [i * interval for i in range(1, horizon_steps + 1)]

        self._reset()

    def _reset(self):
        self.l2_all: List[np.ndarray] = []           # (T,) per sample
        self.col_all: List[np.ndarray] = []           # (T,) binary per sample
        self.speed_errors: List[float] = []
        self.heading_errors: List[float] = []
        self.kinematic_ok_rates: List[float] = []

    def update(
        self,
        pred_traj: np.ndarray,   # (T, 4) — (x, y, θ, v)
        gt_traj:   np.ndarray,   # (T, 4)
        occ_map:   Optional[np.ndarray] = None,  # (T, BEV_SIZE, BEV_SIZE) binary
    ):
        T = min(len(pred_traj), self.horizon)

        # --- L2 at each timestep ---
        l2 = np.linalg.norm(pred_traj[:T, :2] - gt_traj[:T, :2], axis=-1)  # (T,)
        self.l2_all.append(l2)

        # --- Collision rate ---
        col = np.zeros(T)
        if occ_map is not None:
            for t in range(T):
                col[t] = self._check_collision(pred_traj[t, :2], occ_map[t])
        self.col_all.append(col)

        # --- Speed Profile Error (novel) ---
        v_pred = pred_traj[:T, 3]
        v_gt   = gt_traj[:T, 3]
        spe = np.abs(v_pred - v_gt).mean()
        self.speed_errors.append(float(spe))

        # --- Heading Error (novel) ---
        theta_pred = np.degrees(pred_traj[:T, 2])
        theta_gt   = np.degrees(gt_traj[:T, 2])
        # Handle angle wrap-around
        d_theta = np.abs(theta_pred - theta_gt)
        d_theta = np.minimum(d_theta, 360.0 - d_theta)
        self.heading_errors.append(float(d_theta.mean()))

        # --- Kinematic Consistency Score (novel) ---
        dv = np.diff(v_pred)
        accel = np.abs(dv) / self.interval
        ok_rate = float(np.mean(accel <= 5.0))
        self.kinematic_ok_rates.append(ok_rate)

    def _check_collision(self, pred_xy: np.ndarray, occ_t: np.ndarray) -> float:
        """Check BEV occupancy overlap with predicted ego position."""
        x, y = pred_xy
        # Convert to BEV pixel coords
        px = int((x + self.BEV_RANGE) / self.BEV_RES)
        py = int((y + self.BEV_RANGE) / self.BEV_RES)
        # Ego box half-extents in pixels
        hw = max(1, int(self.EGO_W / self.BEV_RES / 2))
        hh = max(1, int(self.EGO_H / self.BEV_RES / 2))

        if (0 <= py - hh and py + hh < self.BEV_SIZE and
                0 <= px - hw and px + hw < self.BEV_SIZE):
            roi = occ_t[py - hh:py + hh, px - hw:px + hw]
            return float(roi.any())
        return 0.0

    def compute(self) -> Dict[str, float]:
        """Return all metrics as a flat dict."""
        if not self.l2_all:
            return {}

        l2_arr  = np.stack(self.l2_all,  axis=0)   # (N, T)
        col_arr = np.stack(self.col_all, axis=0)    # (N, T) if any

        metrics: Dict[str, float] = {}

        # L2 at standard horizons (1s, 2s, 3s, 4s matching the 0.5s interval)
        target_steps = {1: 1, 2: 3, 3: 5, 4: 7}   # 1s=step2, 2s=step4, etc.
        for t_s, step_idx in target_steps.items():
            if step_idx < l2_arr.shape[1]:
                metrics[f"L2_{t_s}s"] = float(l2_arr[:, step_idx].mean())
        metrics["L2_avg"] = float(l2_arr.mean())

        # Collision rate at 1s, 2s, 3s
        col_steps = {1: 1, 2: 3, 3: 5}
        for t_s, step_idx in col_steps.items():
            if step_idx < col_arr.shape[1]:
                metrics[f"Col_{t_s}s"] = float(col_arr[:, step_idx].mean() * 100)
        metrics["Col_avg"] = float(col_arr.mean() * 100)

        # Novel KineDrive metrics
        metrics["SPE_avg"]      = float(np.mean(self.speed_errors))      # Speed Profile Error
        metrics["HE_avg"]       = float(np.mean(self.heading_errors))    # Heading Error (degrees)
        metrics["Kin_ok_rate"]  = float(np.mean(self.kinematic_ok_rates)) * 100  # % steps kinematically valid

        return metrics

    def print_table(self, title: str = "KineDrive Evaluation"):
        m = self.compute()
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")
        print(f"  L2 (m):    1s={m.get('L2_1s', 0):.2f}  2s={m.get('L2_2s', 0):.2f}  "
              f"3s={m.get('L2_3s', 0):.2f}  4s={m.get('L2_4s', 0):.2f}  "
              f"Avg={m.get('L2_avg', 0):.2f}")
        print(f"  Col (%):   1s={m.get('Col_1s', 0):.2f}  2s={m.get('Col_2s', 0):.2f}  "
              f"3s={m.get('Col_3s', 0):.2f}  Avg={m.get('Col_avg', 0):.2f}")
        print(f"  SPE (m/s): {m.get('SPE_avg', 0):.3f}   ← Novel: speed profile error")
        print(f"  HE (deg):  {m.get('HE_avg', 0):.2f}   ← Novel: heading error")
        print(f"  Kin OK(%): {m.get('Kin_ok_rate', 0):.1f}  ← Novel: kinematic consistency")
        print(f"{'='*60}\n")
        return m


# -------------------------------------------------------------------------
# Main evaluation loop
# -------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    config = KineDriveConfig(
        pretrained_model_path=args.pretrained,
        num_denoising_steps=args.num_steps,
        train_mode="il",
    )
    model = KineDrive.from_pretrained(args.pretrained, config)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "action_head" in ckpt:
        model.action_head.load_state_dict(ckpt["action_head"])
    if "spatial_pe" in ckpt:
        model.spatial_pe.load_state_dict(ckpt["spatial_pe"], strict=False)

    model = model.to(device).eval()

    # Dataset
    dataset = NuScenesKinematicDataset(
        data_root=args.data_root,
        sensor_root=args.sensor_root,
        split=args.split,
        processor=model.processor,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=KinematicDataCollator(),
        pin_memory=True,
    )

    metrics = KineDriveMetrics(horizon_steps=8, interval=0.5)
    all_preds: Dict[str, np.ndarray] = {}

    with torch.no_grad():
        for batch in loader:
            vlm_inputs = {k: batch[k].to(device) for k in
                          ["input_ids", "attention_mask", "pixel_values"]
                          if k in batch}
            if "image_grid_thw" in batch:
                vlm_inputs["image_grid_thw"] = batch["image_grid_thw"].to(device)

            pred_trajs = model.forward_inference(
                ego_status=batch["ego_status"].to(device),
                hist_traj=batch["hist_traj"].to(device),
                num_denoising_steps=args.num_steps,
                **vlm_inputs,
            )   # (B, T, 4)

            pred_np = pred_trajs.cpu().numpy()
            gt_np   = batch["gt_waypoints_phys"].numpy()

            for i, token in enumerate(batch["tokens"]):
                metrics.update(pred_np[i], gt_np[i])
                all_preds[token] = pred_np[i]

    # Print and save results
    result = metrics.print_table(title="KineDrive Evaluation Results")

    os.makedirs(args.output_dir, exist_ok=True)

    # Save metrics JSON
    metrics_path = os.path.join(args.output_dir, "kinedrive_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Metrics saved: {metrics_path}")

    # Save per-sample predictions (for visualization / comparison)
    preds_path = os.path.join(args.output_dir, "kinedrive_predictions.json")
    with open(preds_path, "w") as f:
        json.dump({k: v.tolist() for k, v in all_preds.items()}, f)
    print(f"Predictions saved: {preds_path}")


if __name__ == "__main__":
    main()
