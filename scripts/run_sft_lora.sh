#!/bin/bash
# Single-GPU LoRA SFT for AutoVLA (Qwen2.5-VL-3B) on nuScenes.
#
# SMOKE TEST (nuScenes-mini, ~8 train scenes, 1 epoch) -- verify the pipeline runs end to end:
#   bash scripts/run_sft_lora.sh smoke
# FULL RUN (whatever split your config points at):
#   bash scripts/run_sft_lora.sh
#
# Windows PowerShell equivalents are documented in REPRO_LOG.md.

set -e
CONFIG="training/qwen2.5-vl-3B-nusc-sft-lora"

if [ "$1" = "smoke" ]; then
    echo "== SMOKE TEST: 8 train scenes, 1 epoch, 2 val batches =="
    python tools/run_sft_lora.py --config "$CONFIG" \
        --train_sample_size 8 --epochs 1 --limit_val_batches 2
else
    python tools/run_sft_lora.py --config "$CONFIG"
fi
