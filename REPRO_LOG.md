# AutoVLA Reproduction Log

Reproducing **AutoVLA** (NeurIPS 2025, arXiv:2506.13757) — fine-tuning a pretrained
Qwen2.5-VL into an autoregressive trajectory planner and evaluating with real metrics
(PDMS on nuPlan/Navsim; L2 + collision on nuScenes).

- Repo: https://github.com/PeterLoukas/AutoVLA (branch `claude/e2e-vlm-autonomous-driving-fn7f8m`)
- Released checkpoint: HuggingFace `Zewei-Zhou/AutoVLA`
- Base model per repo scripts: `Qwen/Qwen2.5-VL-3B-Instruct` (paper headline uses 7B)

---

## 0. Environments in play (READ THIS FIRST)

There are **two distinct machines**, and it matters which one runs each command:

| Role | Machine | GPU | Can it train/eval AutoVLA? |
|---|---|---|---|
| **Agent container** (where Claude Code runs) | Linux cloud sandbox | **none** | **No** — CPU-only |
| **Your training desktop** | Windows 11 | RTX 5090 32 GB | Yes (with caveats, see §4) |

### Agent container — measured at session start (2026-07-15)
```
GPU:     none (no nvidia-smi, no CUDA)
CPU:     4 cores
RAM:     15 GiB
Disk:    ~30 GiB free (per-session allowance)
Python:  3.11.15  (repo wants 3.9)
conda:   not installed
torch/numpy: not installed
```

**Implication:** the agent container **cannot** download the model/datasets (tens of GB
to TBs), cannot run FSDP training, and cannot run GPU inference/eval. Every command in
Steps 1, 3–8 of the task must run on the **RTX 5090 desktop**. In this container the
agent's job is: read the code, prepare/patch configs & scripts for a single-GPU run,
statically validate the pipeline, and maintain this log. Numbers get filled in from
runs the user executes on the desktop and pastes back.

---

## 1. Method breakdown (from reading the repo source)

Verified by reading: `tools/run_sft.py`, `models/autovla.py`, `models/action_tokenizer.py`,
`dataset_utils/sft_dataset.py`, `tools/eval/nusc_eval.py`, `tools/eval/planning_metrics.py`,
`config/training/*.yaml`, all `scripts/*.sh`, `requirements.txt`, `environment*.yml`.

### 1.1 Architecture
- Backbone: `Qwen2_5_VLForConditionalGeneration` (`models/autovla.py:480`). Vision tower
  **frozen** (`train_vision_backbone: false`), LM backbone **trained**
  (`train_lm_backbone: true`).
- Input prompt (`AutoVLA.get_prompt`): 3 camera views (front / front-left / front-right),
  each as a **4-frame video @ 2 Hz**, plus a text block with current velocity, acceleration,
  and a high-level driving command. `min_pixels == max_pixels == 109760` (≈ 28×28×140 tokens).
- Two prompt modes: `use_cot: true` adds a Chain-of-Thought system prompt + `<think>…</think>`
  reasoning target; `use_cot: false` is action-only.

### 1.2 Action tokenizer (trajectory → tokens)
- `models/action_tokenizer.py`: a **2048-entry codebook** (`codebook_cache/agent_vocab.pkl`,
  already shipped in the repo, 1.18 MB) discretizes short trajectory segments. Each of the
  2048 tokens maps to a `(6,4,2)` local motion primitive.
- New tokens `<action_0>…<action_2047>` are appended to the Qwen tokenizer; `action_start_id
  = 151665`. Trajectory is reconstructed by autoregressive `rollout()` composing primitives
  in global frame.
- Trajectory spec: **10 poses, 0.5 s interval, 5.0 s horizon** (waypoints carry position +
  heading; speed is implicit in spacing).

### 1.3 SFT (`SFTAutoVLA`, `tools/run_sft.py`)
- Loss = Qwen LM cross-entropy on the assistant span only (`DataCollator` masks everything
  before the assistant turn), **plus** an extra CE term on action-token positions
  (`models/autovla.py:336-361`). CoT samples get the LM loss ×40 before adding action loss.
- Trainer: PyTorch-Lightning + **FSDP FULL_SHARD**, bf16 mixed precision, gradient
  checkpointing on, `devices='auto'`. LR 2e-5, AdamW, 5 epochs, batch 1 × grad-accum 4,
  warmup 500 steps. Checkpoints top-3 on `val_loss` to `runs/sft/<timestamp>/`.
- Training data (mix config): `./dataset/nuplan/trainval` + `./dataset/nuscenes/nuscenes_train`.
  Val is nuPlan `navtest`.

### 1.4 RFT / GRPO (`GRPOAutoVLA`, optional)
- Loads the SFT checkpoint as reference model, **LoRA** on q/k/v/o (r=8), GRPO with a
  PDM-score reward (`models/utils/score.py` → Navsim PDM) + a CoT-length penalty. 1 epoch,
  LR 3e-5, needs `devices: [0,1]` in config and the nuPlan metric cache.

### 1.5 Evaluation
- **nuScenes** (`tools/eval/nusc_eval.py`): L2 (0.5–3.0 s) + collision rate vs UniAD-style
  segmentation `.pt` files (external download). Reports STP3 cumulative-avg and UniAD
  per-timestep tables. Needs a trained `.ckpt` + `--seg_data_path`.
- **nuPlan/Navsim PDMS** (`navsim/scripts/evaluation/run_autovla_agent_pdm_score_evaluation.sh`
  → `navsim/navsim/agents/autovla_agent.py`): closed-loop-style PDM score; needs nuPlan maps,
  sensor blobs, metric cache. Set `LORA=false` for the merged HF checkpoint.

### 1.6 Pipeline artifacts (what each step must produce)
| Step | Command | Artifact to verify |
|---|---|---|
| 1 | `scripts/download_qwen.sh` | `./Qwen2.5-VL-3B-Instruct/` (~7 GB) |
| 3 | `scripts/run_nuscenes_preprocessing.sh` | `dataset/nuscenes/nuscenes_{train,val}/*.json` |
| 4 | `scripts/action_token_cluster.sh` | `codebook_cache/agent_vocab.pkl` — **already present** |
| 5 | `scripts/run_sft.sh` | `runs/sft/<ts>/epoch=*.ckpt` |
| 7 | `tools/eval/nusc_eval.py` | `planning_table.txt` (L2 + collision) |
| 8 | eval vs `Zewei-Zhou/AutoVLA` | sanity numbers matching paper |

---

## 2. Codebook sanity check (attempted in agent container)
- `codebook_cache/agent_vocab.pkl` present, 1.18 MB. Structural inspection **blocked**:
  `numpy` not installed in the agent container (bare Python 3.11). Will verify on the
  desktop env instead. Not a concern — the file is version-controlled and used as-is by
  `ActionTokenizer`.

---

## 3. Smallest viable path (proposed)
Given a **single 32 GB GPU** and limited disk, the nuScenes-only, action-only (no-CoT)
path is the smallest thing that yields real published-metric numbers (Avg L2 + Avg
collision). nuPlan/Waymo/CARLA and PDMS need multi-GPU + TB-scale storage and are out of
scope for a first pass. See §4 for the memory blocker that must be resolved first.

---

## 4. Blocker resolved — single-GPU SFT via LoRA
`config/training/*-sft.yaml` sets `train_lm_backbone: true` (full-parameter SFT) and
`run_sft.py` uses FSDP `FULL_SHARD` with `cpu_offload=False`. On a **single** GPU,
FULL_SHARD shards across 1 device → no memory saving. Full-parameter AdamW on the 3B model:

```
params bf16   2B × 3B =  6 GB
grads  bf16   2B × 3B =  6 GB
adam master   4B × 3B = 12 GB   (fp32 copy)
adam m + v    8B × 3B = 24 GB
--------------------------------  ≈ 48 GB  +  activations  →  OOM on 32 GB
```

**Decision (user-approved): LoRA SFT.** Added a single-GPU LoRA path that keeps the model,
data, action codebook, prompts, and loss identical to the paper's SFT, but trains a LoRA
adapter (q/k/v/o) instead of the full backbone. New files:

| File | Purpose |
|---|---|
| `config/training/qwen2.5-vl-3B-nusc-sft-lora.yaml` | nuScenes-only, action-only (no-CoT) LoRA SFT config |
| `tools/run_sft_lora.py` | single-GPU (no FSDP) LoRA trainer with smoke-test CLI overrides |
| `tools/merge_lora.py` | merges the adapter into base weights → eval-ready checkpoint |
| `scripts/run_sft_lora.sh` | launcher (`smoke` arg for the mini smoke test) |

**Critical correctness detail:** a pure attention-only LoRA would leave the 2048 newly
added `<action_*>` token embeddings at random init (they're never in the LoRA target set),
so the model could never emit meaningful actions. `run_sft_lora.py` therefore keeps the
token embedding / lm_head trainable via PEFT `modules_to_save`, auto-deriving the exact set
from `tie_word_embeddings` (tied → `["embed_tokens"]`; untied → `["embed_tokens","lm_head"]`)
to avoid accidentally untying them.

---

## 5. Desktop runbook (RTX 5090) — nuScenes path

Run all of this in the `qwen_finetune` conda env on the desktop, from the `AutoVLA` repo root.
You have **nuScenes-mini** already, so start with the smoke test; download `v1.0-trainval`
only after the mini pipeline is green.

### 5.1 One-time setup
```bash
conda env create -f environment.yml          # creates env "autovla_codeclean" (py3.9)
conda activate autovla_codeclean
pip install -e . --no-warn-conflicts
bash install.sh                              # flash-attn etc. (optional on Windows)
cd navsim && pip install -e . --no-warn-conflicts && cd ..
# LoRA path needs peft (already in requirements via `peft`). Confirm: python -c "import peft"
```

### 5.2 Step 1 — pretrained model
```bash
bash scripts/download_qwen.sh                # -> ./Qwen2.5-VL-3B-Instruct  (~7 GB)
```

### 5.3 Step 3 — preprocess nuScenes-MINI (separate env for nuscenes-devkit)
```bash
conda env create -f environment_nusc_preprocess.yml && conda activate autovla_nusc_preprocess
# NOTE: --output_dir ...\nuscenes  ->  produces  ...\nuscenes_train  and  ...\nuscenes_val
#       which is what config/training/qwen2.5-vl-3B-nusc-sft-lora.yaml points at.
bash scripts/run_nuscenes_preprocessing.sh \
    --nuscenes_path ./dataset/nuscenes \
    --output_dir ./dataset/nuscenes/nuscenes \
    --version v1.0-mini
conda activate autovla_codeclean
# Verify: ls ./dataset/nuscenes/nuscenes_train/*.json | wc -l   (expect ~8 scenes' worth)
```

### 5.4 Step 4 — action codebook
Already shipped: `codebook_cache/agent_vocab.pkl` (1.18 MB). No action needed.

### 5.5 Step 5 — LoRA SFT smoke test, then full
```bash
# Smoke test: 8 scenes, 1 epoch, 2 val batches. Goal = pipeline runs, VRAM fits, loss drops.
bash scripts/run_sft_lora.sh smoke
#   -> checkpoints in runs/sft_lora/<timestamp>/

# Full mini run (all mini_train scenes, 5 epochs):
bash scripts/run_sft_lora.sh

# Windows PowerShell equivalent (if not using Git Bash):
#   python tools\run_sft_lora.py --config training/qwen2.5-vl-3B-nusc-sft-lora `
#       --train_sample_size 8 --epochs 1 --limit_val_batches 2
```
Watch: `nvidia-smi` VRAM (expect << 32 GB with LoRA + grad checkpointing), and that
`print_trainable_parameters()` shows LoRA + embed/lm_head trainable (NOT the whole 3B).

### 5.6 Merge adapter → eval-ready checkpoint
```bash
python tools/merge_lora.py \
    --config config/training/qwen2.5-vl-3B-nusc-sft-lora.yaml \
    --adapter_ckpt runs/sft_lora/<timestamp>/epoch=..-loss=...ckpt \
    --out checkpoints/nusc_sft_lora_merged.ckpt
```

### 5.7 Step 7 — nuScenes eval (L2 + collision)
Download the UniAD-style segmentation `.pt` files first (link in README §nuScenes Evaluation),
then:
```bash
python tools/eval/nusc_eval.py \
    --config config/eval/qwen2.5-vl-3B-nusc-sft-eval.yaml \
    --checkpoint checkpoints/nusc_sft_lora_merged.ckpt \
    --seg_data_path /path/to/nusc_eval_seg \
    --output outputs/planning_table_mini.txt
```
> Note: the eval config's `data.val.sensor_data_path` should be `null` for nuScenes (JSONs
> hold absolute image paths). If you hit "file not found" on images, set it to `null`.

### 5.8 Step 8 — sanity vs released checkpoint
Run 5.7 with `--checkpoint <Zewei-Zhou/AutoVLA merged ckpt>` to confirm the harness reproduces
the paper's numbers before trusting your own. Fill both rows into the table in §7.

### 5.9 Scale up
Once mini is green end-to-end: download `v1.0-trainval`, re-run 5.3 with `--version
v1.0-trainval`, set `train_sample_size: null` (already), and launch `bash scripts/run_sft_lora.sh`.

---

## 6. Deviations from the paper (running list)
1. **LoRA SFT instead of full-parameter SFT** — hardware (single 32 GB GPU). Same data, loss,
   codebook, prompts, targets. Token embeddings/lm_head kept trainable so action tokens learn.
   Expect metrics somewhat below full-parameter SFT; quantify against the released checkpoint.
2. **Base model 3B, not 7B** — per the repo's own scripts (paper headline uses 7B).
3. **nuScenes-only, no-CoT** for the first pass — nuPlan/PDMS + CoT are the heavier follow-ups.
4. **No FSDP** — single-device Lightning strategy; FSDP FULL_SHARD is a no-op on 1 GPU.

## 7. Metrics table (to be filled from desktop runs)
| Model | Avg L2 (m) ↓ | Avg Collision (%) ↓ | Notes |
|---|---|---|---|
| Paper (reported) | _tbd_ | _tbd_ | from arXiv:2506.13757 |
| Released ckpt `Zewei-Zhou/AutoVLA` (our harness) | _tbd_ | _tbd_ | §5.8 sanity |
| Our LoRA SFT (mini) | _tbd_ | _tbd_ | smoke/first pass |
| Our LoRA SFT (trainval) | _tbd_ | _tbd_ | scaled-up |

---

## 8. Actual working environment on the RTX 5090 desktop (2026-07-15)
The repo's pinned `requirements.txt` (torch 2.4.0, numpy 1.23.4, py3.9) is **incompatible with
the RTX 5090 (Blackwell/sm_120)**, which needs torch ≥2.7 + CUDA 12.8. What actually works,
installed into the existing `qwen_finetune` conda env (Python 3.10):

- **torch 2.11.0+cu128 / torchvision 0.26.0+cu128** (kept; `cuda.get_device_capability()==(12,0)`) ✅
- transformers 4.49.0, tokenizers 0.21.4, peft 0.19.1, accelerate 1.14.0
- **pytorch-lightning 2.6.5** (not the repo's 2.2.1 — 2.2.1 predates torch 2.11)
- qwen-vl-utils 0.0.10, safetensors, sentencepiece, einops, torchmetrics, tensorboard
- geo stack via **conda-forge** (libmamba solver): geopandas, fiona, rasterio, shapely, rtree, pyproj
- pure-python via pip: hydra-core 1.2.0, omegaconf, pyquaternion, pyarrow, ujson, nest-asyncio, retry
- **nuplan-devkit 1.2.0** installed **editable from a local git clone** (`pip install -e . --no-deps
  --no-build-isolation`) — the PyPI/wheel build fails on Windows with a `build\lib\docs` collision.
- navsim 1.1.0 editable (`pip install -e .\navsim --no-deps`)
- Import-chain fills needed by nuplan on Windows: **fcntl shim** (`tools/windows_shims/fcntl.py` →
  site-packages), aioboto3/aiobotocore/boto3/botocore/s3transfer, pytest, opencv-python,
  positional-encodings.

Install strategy: **never `pip install -r requirements.txt`** (it would downgrade torch and break
the GPU). Install `-e . --no-deps`, then curated deps that don't pin torch/numpy. The giant pip
"dependency conflicts" wall about `autovla requires torch==2.4.0 ...` is expected and harmless.

Import smoke test passing (the GATE-3 gate):
```
python -c "import sys; sys.path.insert(0,'navsim'); from dataset_utils.sft_dataset import SFTDataset; from models.autovla import SFTAutoVLA; from navsim.agents.autovla_agent import AutoVLAAgent; print('IMPORTS OK')"
# -> IMPORTS OK
```

**Platform caveat:** these Windows shims cover the nuScenes **open-loop** path (L2 + collision).
The nuPlan **closed-loop PDMS** path needs real nuplan map/sim code and effectively requires
**Linux/WSL2**; plan to move there for the PDMS phase.

## Command / run journal
(Every command executed on the desktop, with output, VRAM, and wall-clock, gets appended
below as the reproduction proceeds.)

| Date | Machine | Command | Result | VRAM | Wall-clock |
|---|---|---|---|---|---|
| 2026-07-15 | agent container | env probe (nvidia-smi/free/df/python) | no GPU, 4 CPU, 15 GiB RAM, 30 GiB disk | — | — |
