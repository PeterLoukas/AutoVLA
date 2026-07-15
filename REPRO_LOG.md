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

## 4. Open blocker — single-GPU SFT does not fit as written
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

Resolution options (decision needed from user before writing the SFT config):
1. **LoRA SFT** — smallest footprint, fits easily; requires adding a LoRA wrap to the SFT
   path (RFT already uses LoRA, so the pattern exists in-repo). Fastest to green.
2. **FSDP `cpu_offload=True`** — keeps full-parameter fidelity to the paper; offloads to the
   64 GB DDR5 host RAM; much slower per step but truest reproduction.
3. **QLoRA (4-bit base)** — lowest VRAM, but adds bitsandbytes and deviates more from paper.

---

## Command / run journal
(Every command executed on the desktop, with output, VRAM, and wall-clock, gets appended
below as the reproduction proceeds.)

| Date | Machine | Command | Result | VRAM | Wall-clock |
|---|---|---|---|---|---|
| 2026-07-15 | agent container | env probe (nvidia-smi/free/df/python) | no GPU, 4 CPU, 15 GiB RAM, 30 GiB disk | — | — |
