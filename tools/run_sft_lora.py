"""
Single-GPU LoRA SFT for AutoVLA (Qwen2.5-VL-3B).

This is a hardware-constrained variant of tools/run_sft.py. It keeps the AutoVLA model,
data pipeline, action codebook, prompts, and loss EXACTLY as the paper's SFT, but:

  * freezes the full backbone and trains a LoRA adapter on the attention projections
    (fits on a single 32 GB GPU, where full-parameter FSDP SFT would OOM), and
  * keeps the token embedding / lm_head trainable so the 2048 newly added <action_*>
    tokens can actually be learned (a pure attention-only LoRA would leave those new
    embedding rows at their random init).

Checkpoints saved here are LoRA-wrapped. Merge them with tools/merge_lora.py before
running tools/eval/nusc_eval.py.

Example (full run):
    python tools/run_sft_lora.py --config training/qwen2.5-vl-3B-nusc-sft-lora

Example (smoke test on nuScenes-mini):
    python tools/run_sft_lora.py --config training/qwen2.5-vl-3B-nusc-sft-lora \
        --train_sample_size 8 --epochs 1 --limit_val_batches 2
"""
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "navsim"))

import yaml
import torch
import argparse
import datetime
import pytorch_lightning as pl

from peft import get_peft_model, LoraConfig, TaskType
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

from dataset_utils.sft_dataset import SFTDataset, DataCollator
from models.autovla import SFTAutoVLA
from transformers import AutoProcessor

torch.set_float32_matmul_precision('high')


def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def derive_modules_to_save(vlm, config_modules_to_save):
    """Decide which full modules to keep trainable so the new action tokens can learn.

    If the user pinned modules_to_save in the config, respect it. Otherwise auto-derive
    from tie_word_embeddings: when embeddings are tied, saving only 'embed_tokens' also
    updates the lm_head (they share a tensor); listing both would UNTIE them.
    """
    if config_modules_to_save:
        return list(config_modules_to_save)
    tied = bool(getattr(vlm.config, "tie_word_embeddings", False))
    return ["embed_tokens"] if tied else ["embed_tokens", "lm_head"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, required=True)
    # Smoke-test / override knobs (do not require editing the YAML)
    parser.add_argument("--train_sample_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--precision", type=str, default="bf16-true")
    args = parser.parse_args()
    seed_everything(args.seed)

    config = load_config(f"./config/{args.config}.yaml")

    # Allow CLI to override a couple of training knobs for smoke tests
    if args.train_sample_size is not None:
        config['training']['train_sample_size'] = args.train_sample_size
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs

    processor = AutoProcessor.from_pretrained(config['model']['pretrained_model_path'], use_fast=True)
    using_cot = config['model']['use_cot']

    train_dataset = SFTDataset(config['data']['train'], config['model'], processor, using_cot=using_cot)

    train_sample_size = config['training']['train_sample_size']
    if train_sample_size is not None and len(train_dataset) > train_sample_size:
        print(f"Subsampling train set to {train_sample_size} of {len(train_dataset)} scenes")
        indices = torch.randperm(len(train_dataset))[:train_sample_size]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    else:
        print(f"Using full train set ({len(train_dataset)} scenes)")

    val_dataset = SFTDataset(config['data']['val'], config['model'], processor, using_cot=using_cot)

    model = SFTAutoVLA(config)

    # === Apply LoRA (mirrors the in-repo pattern from tools/run_rft.py) ===
    lora_conf = config['model'].get('lora', {})
    assert lora_conf.get('use', False), "This script expects model.lora.use: true in the config"
    modules_to_save = derive_modules_to_save(model.autovla.vlm, lora_conf.get('modules_to_save'))
    lora_config = LoraConfig(
        task_type=TaskType[lora_conf.get("task_type", "CAUSAL_LM")],
        target_modules=lora_conf.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        r=lora_conf.get("r", 16),
        lora_alpha=lora_conf.get("alpha", 32),
        lora_dropout=lora_conf.get("dropout", 0.05),
        bias=lora_conf.get("bias", "none"),
        modules_to_save=modules_to_save,
    )
    print(f"LoRA target_modules={lora_config.target_modules}, modules_to_save={modules_to_save} "
          f"(tie_word_embeddings={getattr(model.autovla.vlm.config, 'tie_word_embeddings', None)})")
    model.autovla.vlm = get_peft_model(model.autovla.vlm, lora_config)
    model.autovla.vlm.print_trainable_parameters()

    # Gradient checkpointing for PEFT: inputs to checkpointed blocks must require grad,
    # and KV cache must be off. enable_input_require_grads() is the standard PEFT hook.
    model.autovla.vlm.config.use_cache = False
    model.autovla.vlm.enable_input_require_grads()
    model.autovla.vlm.gradient_checkpointing_enable()

    data_collator = DataCollator(
        processor=processor,
        ignore_index=config['model']['tokens']['ignore_index'],
        assistant_id=config['model']['tokens']['assistant_id'],
    )

    train_data = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        collate_fn=data_collator,
        num_workers=config['training']['num_workers'],
        shuffle=True,
    )
    val_data = DataLoader(
        val_dataset,
        batch_size=config['inference']['batch_size'],
        collate_fn=data_collator,
        num_workers=config['inference']['num_workers'],
        shuffle=False,
    )

    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"runs/sft_lora/{current_date}"

    trainer = pl.Trainer(
        num_nodes=1,
        max_epochs=config['training']['epochs'],
        max_steps=args.max_steps,
        accelerator="gpu",
        devices=1,                       # single-GPU: no FSDP sharding needed
        strategy="auto",
        precision=args.precision,
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        limit_val_batches=args.limit_val_batches,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                dirpath=f"{save_dir}",
                filename="epoch={epoch}-loss={val_loss:.4f}",
                auto_insert_metric_name=False,
                save_weights_only=True,
                every_n_epochs=1,
            ),
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            LearningRateMonitor(logging_interval="step"),
        ],
        gradient_clip_algorithm='value',
        gradient_clip_val=1.0,
        logger=CSVLogger(save_dir=f"{save_dir}"),
        enable_model_summary=True,
    )
    torch.cuda.empty_cache()
    print(f"Saving checkpoints to: {save_dir}")
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
