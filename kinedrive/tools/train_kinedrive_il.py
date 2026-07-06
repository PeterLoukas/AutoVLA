"""
KineDrive Stage 2: Imitation Learning with Flow Matching

Trains the KinematicFlowMatchingHead to denoise 4D trajectories (x, y, θ, v)
conditioned on frozen VLM hidden states.

This stage follows ReCogDrive's IL pipeline but extends to 4D kinematic space:
  - Flow matching instead of DDPM (fewer steps, more stable)
  - 4D action space adds explicit speed supervision
  - Kinematic consistency loss penalizes physically implausible speed profiles

Usage:
  torchrun --nproc_per_node=8 kinedrive/tools/train_kinedrive_il.py \
      --config kinedrive/configs/kinedrive_il.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from kinedrive.models.kinedrive import KineDrive, KineDriveConfig
from kinedrive.datasets.nuscenes_kinematic import NuScenesKinematicDataset, KinematicDataCollator


def parse_args():
    parser = argparse.ArgumentParser(description="KineDrive IL Training")
    parser.add_argument("--pretrained", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--sft_checkpoint", default=None, help="Path to Stage 1 SFT checkpoint")
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--sensor_root", required=True)
    parser.add_argument("--output_dir",  default="./checkpoints/kinedrive_il")
    parser.add_argument("--cot_path",    default=None)
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--grad_accum",  type=int,   default=1)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--num_denoising_steps", type=int, default=10)
    parser.add_argument("--save_every",  type=int,   default=5)
    parser.add_argument("--local_rank",  type=int,   default=-1)
    return parser.parse_args()


def train_epoch(model, loader, optimizer, scaler, epoch, rank, grad_accum):
    model.train()
    total_loss = 0.0
    total_flow = 0.0
    total_kin  = 0.0
    steps = 0

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        device = next(model.parameters()).device

        gt_wp    = batch["gt_waypoints_phys"].to(device)
        ego_st   = batch["ego_status"].to(device)
        hist_tr  = batch["hist_traj"].to(device)
        cur_spd  = batch["current_speed"].to(device)

        # VLM inputs (for hidden state extraction)
        vlm_kwargs = {}
        for k in ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]:
            if k in batch:
                vlm_kwargs[k] = batch[k].to(device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            losses = model.forward_il(
                gt_waypoints_phys=gt_wp,
                ego_status=ego_st,
                hist_traj=hist_tr,
                current_speed=cur_spd,
                **vlm_kwargs,
            )

        loss = losses["loss_total"] / grad_accum
        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += losses["loss_total"].item()
        total_flow += losses["loss_flow"].item()
        total_kin  += losses["loss_kin"].item()
        steps += 1

        if rank == 0 and step % 100 == 0:
            print(
                f"[Epoch {epoch:03d} Step {step:05d}] "
                f"loss={losses['loss_total']:.4f} "
                f"flow={losses['loss_flow']:.4f} "
                f"kin={losses['loss_kin']:.4f}"
            )

    return {
        "loss":      total_loss / max(steps, 1),
        "loss_flow": total_flow / max(steps, 1),
        "loss_kin":  total_kin  / max(steps, 1),
    }


def main():
    args = parse_args()

    # DDP setup
    if args.local_rank == -1:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{args.local_rank}")
        torch.cuda.set_device(device)

    # Build model
    config = KineDriveConfig(
        pretrained_model_path=args.pretrained,
        num_denoising_steps=args.num_denoising_steps,
        train_mode="il",
    )
    model = KineDrive.from_pretrained(args.pretrained, config)

    # Load SFT checkpoint (Stage 1 → Stage 2)
    if args.sft_checkpoint and os.path.exists(args.sft_checkpoint):
        if rank == 0:
            print(f"Loading SFT checkpoint: {args.sft_checkpoint}")
        state = torch.load(args.sft_checkpoint, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if rank == 0:
            print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # Freeze VLM — only train the action head (and 3D PE)
    for name, param in model.named_parameters():
        if "action_head" not in name and "spatial_pe" not in name:
            param.requires_grad = False

    model = model.to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)

    # Dataset
    dataset = NuScenesKinematicDataset(
        data_root=args.data_root,
        sensor_root=args.sensor_root,
        split="train",
        processor=model.module.processor if world_size > 1 else model.processor,
        cot_path=args.cot_path,
    )
    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        collate_fn=KinematicDataCollator(),
        pin_memory=True,
    )

    # Optimizer — only tune action head + 3D PE
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader), eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler()

    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        metrics = train_epoch(
            model, loader, optimizer, scaler,
            epoch, rank, args.grad_accum,
        )
        scheduler.step()

        if rank == 0:
            print(f"[Epoch {epoch:03d}] {metrics}")

            if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
                m = model.module if world_size > 1 else model
                ckpt = {
                    "epoch":          epoch,
                    "action_head":    m.action_head.state_dict(),
                    "spatial_pe":     m.spatial_pe.state_dict(),
                    "waypoint_space": m.waypoint_space.state_dict(),
                    "config":         config.__dict__,
                }
                path = os.path.join(args.output_dir, f"kinedrive_il_epoch{epoch:03d}.pt")
                torch.save(ckpt, path)
                print(f"  → Saved checkpoint: {path}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
