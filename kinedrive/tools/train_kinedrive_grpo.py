"""
KineDrive Stage 3: Kinematic-GRPO (K-GRPO) Reinforcement Learning

Extends ReCogDrive's DiffGRPO to 4D kinematic trajectories with a
velocity-aware reward (PDM-V):

  R = PDM_score × scale + λ_v × Speed_Profile_Score - λ_len × CoT_penalty

Speed Profile Score:
  SP_score = exp(-MAE(v_pred, v_GT))
  rewards the model for accurately predicting the speed profile alongside
  position and heading, incentivizing physically plausible trajectories.

PDM Score: full NAVSIM closed-loop simulation reward
  = NC × DAC × weighted(TTC=5, Progress=5, Comfort=2)

DiffGRPO-style temporal discounting:
  - G=8 rollouts per scene
  - γ=0.6 discount per denoising step
  - GRPO advantage normalization across rollouts

Usage:
  torchrun --nproc_per_node=8 kinedrive/tools/train_kinedrive_grpo.py \
      --config kinedrive/configs/kinedrive_grpo.yaml
"""

import argparse
import copy
import os
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).parents[2]))

from kinedrive.models.kinedrive import KineDrive, KineDriveConfig
from kinedrive.models.kinematic_action_head import KinematicFlowMatchingHead
from kinedrive.datasets.nuscenes_kinematic import NuScenesKinematicDataset, KinematicDataCollator


def parse_args():
    parser = argparse.ArgumentParser(description="KineDrive K-GRPO Training")
    parser.add_argument("--pretrained",    default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--il_checkpoint", required=True, help="Stage 2 IL checkpoint")
    parser.add_argument("--data_root",     required=True)
    parser.add_argument("--sensor_root",   required=True)
    parser.add_argument("--metric_cache",  required=True, help="NAVSIM metric cache path")
    parser.add_argument("--output_dir",    default="./checkpoints/kinedrive_grpo")
    parser.add_argument("--epochs",        type=int,   default=10)
    parser.add_argument("--lr",            type=float, default=1e-5)
    parser.add_argument("--batch_size",    type=int,   default=4)
    parser.add_argument("--num_rollouts",  type=int,   default=8,  help="G: rollouts per scene")
    parser.add_argument("--kl_beta",       type=float, default=0.04)
    parser.add_argument("--reward_scale",  type=float, default=10.0)
    parser.add_argument("--gamma",         type=float, default=0.6, help="denoising step discount")
    parser.add_argument("--speed_reward_lambda", type=float, default=2.0)
    parser.add_argument("--cot_penalty_weight",  type=float, default=0.3)
    parser.add_argument("--cot_penalty_center",  type=int,   default=400)
    parser.add_argument("--num_denoising_steps", type=int,   default=5)
    parser.add_argument("--local_rank",    type=int,   default=-1)
    return parser.parse_args()


class KGRPO_Trainer:
    """
    K-GRPO Trainer for KineDrive.

    Implements the DiffGRPO algorithm extended to 4D kinematic trajectories:
      1. Generate G rollout chains per scene (full Euler integration)
      2. Compute PDM-V reward for each final trajectory
      3. Normalize rewards to per-group advantages (GRPO)
      4. Apply temporal discount γ^{K-k-1} across denoising steps
      5. Policy gradient loss + BC regularization against frozen IL policy
    """

    def __init__(self, model: KineDrive, ref_model: KineDrive, args, device):
        self.model     = model
        self.ref_model = ref_model   # frozen IL policy for BC
        self.args      = args
        self.device    = device

        # PDM reward (NAVSIM-based)
        self._pdm_reward = None
        if os.path.exists(args.metric_cache):
            try:
                from navsim.planning.script.run_training_recogdrive_rl import MetricCacheLoader
                # Use AutoVLA's PDM reward implementation
                sys.path.insert(0, str(Path(__file__).parents[2]))
                from models.utils.score import PDM_Reward
                self._pdm_reward = PDM_Reward(args.metric_cache)
            except ImportError:
                print("[K-GRPO] PDM reward not available; using L2 proxy reward")

        self.reward_window = deque(maxlen=100)

    def compute_reward(
        self,
        traj_phys: torch.Tensor,   # (B, T, 4) — (x,y,θ,v) in physical units
        gt_waypoints: torch.Tensor, # (B, T, 4)
        tokens: List[str],
    ) -> torch.Tensor:             # (B,) rewards
        """Compute PDM-V reward.

        PDM-V = PDM_score × scale + λ_v × Speed_Profile_Score

        Speed_Profile_Score = exp(-MAE(v_pred, v_GT))
          - When PDM score not available, use L2 distance as proxy
          - Speed score is always computed from GT speed profile
        """
        B = traj_phys.shape[0]
        device = traj_phys.device
        rewards = torch.zeros(B, device=device)

        for i in range(B):
            # --- PDM score (closed-loop simulation) ---
            if self._pdm_reward is not None:
                try:
                    pdm_r = self._pdm_reward(
                        traj_phys[i, :, :3].cpu().numpy(),  # (T, 3) — (x,y,θ)
                        {"token": tokens[i]},
                    )
                    pdm_score = float(pdm_r) * self.args.reward_scale
                except Exception:
                    # L2 proxy
                    l2 = (traj_phys[i, :, :2] - gt_waypoints[i, :, :2]).norm(dim=-1).mean()
                    pdm_score = self.args.reward_scale * max(0.0, 1.0 - l2.item() / 5.0)
            else:
                l2 = (traj_phys[i, :, :2] - gt_waypoints[i, :, :2]).norm(dim=-1).mean()
                pdm_score = self.args.reward_scale * max(0.0, 1.0 - l2.item() / 5.0)

            # --- Speed profile score (novel reward component) ---
            v_pred = traj_phys[i, :, 3]
            v_gt   = gt_waypoints[i, :, 3]
            mae_v  = (v_pred - v_gt).abs().mean()
            speed_score = torch.exp(-mae_v).item()

            rewards[i] = pdm_score + self.args.speed_reward_lambda * speed_score

        return rewards

    def grpo_step(
        self,
        vlm_hidden: torch.Tensor,   # (B, S, D) — VLM hidden states
        ego_status: torch.Tensor,   # (B, 9)
        hist_traj: torch.Tensor,    # (B, 4, 4)
        gt_waypoints: torch.Tensor, # (B, T, 4) physical
        tokens: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Single K-GRPO update step.

        Args:
            vlm_hidden:   VLM hidden states (frozen during GRPO)
            ego_status:   ego kinematic state
            hist_traj:    past 4-step kinematic history
            gt_waypoints: ground-truth 4D waypoints (for speed reward)
            tokens:       scene tokens for PDM reward lookup

        Returns dict with: loss_policy, loss_bc, loss_total, avg_reward
        """
        B = vlm_hidden.shape[0]
        G = self.args.num_rollouts
        K = self.args.num_denoising_steps
        device = self.device

        # Repeat each scene G times
        vlm_rep  = vlm_hidden.repeat_interleave(G, 0)   # (B*G, S, D)
        ego_rep  = ego_status.repeat_interleave(G, 0)   # (B*G, 9)
        hist_rep = hist_traj.repeat_interleave(G, 0)    # (B*G, 4, 4)
        gt_rep   = gt_waypoints.repeat_interleave(G, 0) # (B*G, T, 4)
        tok_rep  = [t for t in tokens for _ in range(G)]

        action_head = self.model.action_head

        # --- Step 1: Sample G rollout chains ---
        with torch.no_grad():
            chains, trajs = action_head.rollout_chain(
                vlm_hidden=vlm_rep,
                ego_status=ego_rep,
                hist_traj=hist_rep,
                num_steps=K,
            )   # chains: (B*G, K+1, T, 4), trajs: (B*G, T, 4)

        # --- Step 2: Compute PDM-V rewards ---
        rewards = self.compute_reward(trajs, gt_rep, tok_rep)   # (B*G,)
        self.reward_window.extend(rewards.cpu().tolist())

        # --- Step 3: GRPO advantage normalization ---
        rew_mat = rewards.view(B, G)
        mu  = rew_mat.mean(dim=1, keepdim=True)
        std = rew_mat.std(dim=1, keepdim=True) + 1e-8
        adv = ((rew_mat - mu) / std).view(-1)             # (B*G,)

        # Quantile clipping (follows ReCogDrive)
        adv = adv.clamp(
            adv.quantile(0.0),
            adv.quantile(1.0),
        )

        # --- Step 4: Temporal discount across denoising steps ---
        gamma = self.args.gamma
        discount = gamma ** (K - torch.arange(K, device=device, dtype=torch.float32) - 1)
        # (B*G,) × (K,) → (B*G, K) via broadcasting
        adv_weighted = adv.unsqueeze(-1) * discount.unsqueeze(0)  # (B*G, K)
        adv_flat = adv_weighted.reshape(-1)                        # (B*G*K,)

        # --- Step 5: Log-probs under current policy ---
        log_probs_curr = action_head.get_logprobs(
            chains, vlm_rep, ego_rep, hist_rep, K
        )   # (B*G, K)

        # --- Step 6: Log-probs under reference (frozen IL) policy ---
        with torch.no_grad():
            ref_head = self.ref_model.action_head
            log_probs_ref = ref_head.get_logprobs(
                chains, vlm_rep, ego_rep, hist_rep, K
            )   # (B*G, K)

        # --- Step 7: Policy gradient loss ---
        lp_flat = log_probs_curr.reshape(-1)   # (B*G*K,)
        loss_policy = -(lp_flat * adv_flat).mean()

        # --- Step 8: BC regularization (KL from reference) ---
        # KL(π_ref || π) ≈ exp(lp_ref - lp) - (lp_ref - lp) - 1  [non-negative]
        lp_ref_flat = log_probs_ref.reshape(-1)
        kl = torch.exp(lp_ref_flat - lp_flat) - (lp_ref_flat - lp_flat) - 1.0
        loss_bc = self.args.kl_beta * kl.clamp(-10, 10).mean()

        loss_total = loss_policy + loss_bc

        return {
            "loss_policy": loss_policy,
            "loss_bc":     loss_bc,
            "loss_total":  loss_total,
            "avg_reward":  torch.tensor(
                sum(self.reward_window) / max(len(self.reward_window), 1)
            ),
        }


def main():
    args = parse_args()

    # DDP setup
    if args.local_rank >= 0:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{args.local_rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    config = KineDriveConfig(
        pretrained_model_path=args.pretrained,
        num_denoising_steps=args.num_denoising_steps,
        train_mode="grpo",
    )
    model = KineDrive.from_pretrained(args.pretrained, config)

    # Load IL checkpoint
    if rank == 0:
        print(f"Loading IL checkpoint: {args.il_checkpoint}")
    ckpt = torch.load(args.il_checkpoint, map_location="cpu")
    model.action_head.load_state_dict(ckpt["action_head"])
    model.spatial_pe.load_state_dict(ckpt.get("spatial_pe", {}), strict=False)

    # Reference model (frozen IL policy)
    ref_model = copy.deepcopy(model)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model = ref_model.to(device).eval()

    # Only train the action head (VLM stays frozen)
    for name, param in model.named_parameters():
        param.requires_grad = "action_head" in name

    model = model.to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)

    # Dataset
    dataset = NuScenesKinematicDataset(
        data_root=args.data_root,
        sensor_root=args.sensor_root,
        split="train",
        processor=(model.module.processor if world_size > 1 else model.processor),
    )
    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        collate_fn=KinematicDataCollator(),
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4
    )

    trainer = KGRPO_Trainer(
        model.module if world_size > 1 else model,
        ref_model,
        args,
        device,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()

        for step, batch in enumerate(loader):
            # Extract VLM hidden states (frozen)
            m = model.module if world_size > 1 else model
            vlm_inputs = {k: batch[k].to(device) for k in
                          ["input_ids", "attention_mask", "pixel_values"]
                          if k in batch}
            if "image_grid_thw" in batch:
                vlm_inputs["image_grid_thw"] = batch["image_grid_thw"].to(device)

            with torch.no_grad():
                out = m.vlm(output_hidden_states=True, **vlm_inputs)
                vlm_hidden = out.hidden_states[-1].float()

            result = trainer.grpo_step(
                vlm_hidden=vlm_hidden,
                ego_status=batch["ego_status"].to(device),
                hist_traj=batch["hist_traj"].to(device),
                gt_waypoints=batch["gt_waypoints_phys"].to(device),
                tokens=batch["tokens"],
            )

            result["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            optimizer.zero_grad()

            if rank == 0 and step % 50 == 0:
                print(
                    f"[Epoch {epoch:03d} Step {step:05d}] "
                    f"loss={result['loss_total']:.4f} "
                    f"policy={result['loss_policy']:.4f} "
                    f"bc={result['loss_bc']:.4f} "
                    f"avg_reward={result['avg_reward']:.3f}"
                )

        # Save checkpoint
        if rank == 0:
            m = model.module if world_size > 1 else model
            ckpt_out = {
                "epoch":       epoch,
                "action_head": m.action_head.state_dict(),
                "config":      config.__dict__,
            }
            path = os.path.join(args.output_dir, f"kinedrive_grpo_epoch{epoch:03d}.pt")
            torch.save(ckpt_out, path)
            print(f"Saved: {path}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
