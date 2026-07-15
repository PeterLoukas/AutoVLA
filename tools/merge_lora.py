"""
Merge a LoRA-SFT checkpoint (from tools/run_sft_lora.py) into the base Qwen2.5-VL
weights and write a plain checkpoint that tools/eval/nusc_eval.py can load directly.

nusc_eval.py builds a vanilla SFTAutoVLA(config) and calls
    model.autovla.load_state_dict(state_dict, strict=False)
so we save {'state_dict': autovla.state_dict()} with keys relative to `autovla`
(i.e. 'vlm.model.*', 'vlm.lm_head.*', 'vlm.visual.*') after merging the adapter.

Example:
    python tools/merge_lora.py \
        --config config/training/qwen2.5-vl-3B-nusc-sft-lora.yaml \
        --adapter_ckpt runs/sft_lora/<ts>/epoch=..-loss=...ckpt \
        --out checkpoints/nusc_sft_lora_merged.ckpt
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "navsim"))

import yaml
import torch
import argparse
from peft import get_peft_model, LoraConfig, TaskType

from models.autovla import SFTAutoVLA


def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def derive_modules_to_save(vlm, config_modules_to_save):
    if config_modules_to_save:
        return list(config_modules_to_save)
    tied = bool(getattr(vlm.config, "tie_word_embeddings", False))
    return ["embed_tokens"] if tied else ["embed_tokens", "lm_head"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--adapter_ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config = load_config(args.config)

    # Rebuild the exact LoRA-wrapped structure the checkpoint was saved from
    model = SFTAutoVLA(config)
    lora_conf = config['model'].get('lora', {})
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
    model.autovla.vlm = get_peft_model(model.autovla.vlm, lora_config)

    print(f"Loading adapter checkpoint: {args.adapter_ckpt}")
    ckpt = torch.load(args.adapter_ckpt, map_location=args.device)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # LoRA/base keys should all match; report anything surprising
    real_missing = [k for k in missing if 'lora_' in k or 'modules_to_save' in k]
    if real_missing:
        print(f"WARNING: {len(real_missing)} adapter keys were missing, e.g. {real_missing[:3]}")
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")

    print("Merging LoRA into base weights (merge_and_unload)...")
    model.autovla.vlm = model.autovla.vlm.merge_and_unload()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.autovla.state_dict()}, out_path)
    print(f"Saved merged, eval-ready checkpoint to: {out_path}")
    print("Run eval with tools/eval/nusc_eval.py --checkpoint " + str(out_path))


if __name__ == '__main__':
    main()
