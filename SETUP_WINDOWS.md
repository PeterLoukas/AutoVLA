# AutoVLA on Windows + RTX 5090 — Setup & Run Guide

Target machine: Windows 11, RTX 5090 (32 GB, Blackwell / sm_120), conda env `qwen_finetune`
(Python 3.10, torch 2.11.0+cu128), working dir `C:\Users\user\projects\qwen_architecture`.
Shell: **PowerShell** (so we call `python ...` directly, not `bash scripts/*.sh`).

> Work through this **one GATE at a time**. Each gate ends with a check + what to paste back
> if it fails. Do not move on until a gate's check passes. The hard gate is **GATE 3**
> (nuPlan/navsim deps on Windows) — expect to iterate there.

---

## Why we don't just `pip install -r requirements.txt`
- The repo pins `torch==2.4.0` (CUDA 12.1). Your **RTX 5090 (sm_120) needs torch ≥2.7 / cu128**
  — you already have `torch 2.11.0+cu128`, which is correct. A plain install would **downgrade
  torch and break GPU support.** We preserve your torch.
- `navsim` → `nuplan-devkit` pins `numpy==1.23.4` and a GDAL-based geo stack. On Windows we
  install those from **conda-forge** (which ships GDAL binaries) and install nuplan/navsim with
  `--no-deps` so pip doesn't drag torch/numpy back down.

---

## GATE 0 — Verify the GPU is usable (30 seconds)
```powershell
conda activate qwen_finetune
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
```
**Pass if:** prints `True`, `NVIDIA GeForce RTX 5090`, and capability `(12, 0)` (sm_120).
**If `False` or capability error:** your torch build doesn't support Blackwell — stop and paste
the output; we fix torch before anything else.

---

## GATE 1 — Get the code into your workspace
```powershell
# If `git` is not recognized, install it into the env first:
conda install -c conda-forge git -y
git --version

cd C:\Users\user\projects\qwen_architecture
git clone -b claude/e2e-vlm-autonomous-driving-fn7f8m https://github.com/PeterLoukas/AutoVLA.git
cd AutoVLA
git log --oneline -3
```
> Create the GATE 2 junction only AFTER the repo is cloned, from INSIDE `AutoVLA`. To remove a
> junction safely use `cmd /c "rmdir <path>"` — that deletes only the link, never the target data.
> (Never `Remove-Item -Recurse` a folder that still contains a junction — it can follow the link
> into your real data.)
**Pass if:** you see the commits including `Add single-GPU LoRA SFT path for nuScenes reproduction`.
Your layout is now:
```
qwen_architecture\
├── AutoVLA\            <- the repo (run ALL commands from here)
├── datasets\nuscenes\  <- your nuScenes mini
├── checkpoints\  outputs\  scripts\   (your originals; unused by the repo)
```

---

## GATE 2 — Wire your nuScenes data into the repo layout
The configs use the relative path `./dataset/nuscenes`. Point it at your existing data with a
directory **junction** (no admin needed on Windows):
```powershell
# from qwen_architecture\AutoVLA
New-Item -ItemType Directory -Force -Path .\dataset | Out-Null
New-Item -ItemType Junction -Path .\dataset\nuscenes -Target C:\Users\user\projects\qwen_architecture\datasets\nuscenes
# verify the nuScenes-mini structure is visible through the junction:
dir .\dataset\nuscenes            # expect: maps, samples, sweeps, v1.0-mini
dir .\dataset\nuscenes\v1.0-mini  # expect: category.json, sample.json, scene.json, ...
```
**Pass if:** `v1.0-mini\` and `samples\` are listed.
**If `v1.0-mini` is missing:** your mini download isn't at `datasets\nuscenes`. Paste
`dir C:\Users\user\projects\qwen_architecture\datasets\nuscenes` so we find the real root.

---

## GATE 3 — Install dependencies (the hard one; do it in stages)

### 3a. Register the AutoVLA package without touching torch/numpy
```powershell
# from qwen_architecture\AutoVLA
pip install -e . --no-deps
```

### 3b. Core ML deps that are safe with torch 2.11 (no torch pin) — single line
```powershell
pip install "transformers==4.49.0" "qwen-vl-utils==0.0.10" peft accelerate "pytorch-lightning>=2.3" torchmetrics tensorboard prettytable einops sentencepiece safetensors
```
> We use `pytorch-lightning>=2.3` (not the repo's 2.2.1) because 2.2.1 predates torch 2.11.
> The LoRA trainer only uses stable Lightning APIs, so a newer version is fine.

### 3c. The nuPlan/navsim geo stack via conda-forge (ships GDAL; avoids Windows pip pain) — single line
```powershell
conda install -c conda-forge geopandas fiona rasterio shapely rtree pyproj hydra-core=1.2.0 omegaconf pyquaternion pyarrow joblib retry ujson nest-asyncio -y
```
> Note: `casadi`, `control`, `bokeh`, `ray` are only needed for nuPlan *simulation/PDMS*, not the
> nuScenes import path. We leave them out for now and add on demand if the GATE-3e smoke test asks.

### 3d. Install nuplan-devkit and navsim WITHOUT letting pip change torch/numpy
```powershell
pip install "git+https://github.com/motional/nuplan-devkit@nuplan-devkit-v1.2" --no-deps
pip install -e .\navsim --no-deps
```

### 3e. GATE CHECK — the import smoke test (validates the whole stack, no weights/data needed)
```powershell
python -c "import sys; sys.path.insert(0,'navsim'); from dataset_utils.sft_dataset import SFTDataset; from models.autovla import SFTAutoVLA; from navsim.agents.autovla_agent import AutoVLAAgent; print('IMPORTS OK')"
```
**Pass if:** it prints `IMPORTS OK`.
**If it raises `ModuleNotFoundError: X`:** install `X` (prefer `conda install -c conda-forge X`,
fall back to `pip install X --no-deps`), re-run, repeat. Paste the first traceback if you get
stuck — this is the expected iteration point, and I'll give you the exact fix per module.

**Windows-only stdlib shims.** nuplan-devkit imports Unix-only stdlib modules (starting with
`fcntl`) that don't exist on Windows. The nuScenes path never uses them at runtime, so we drop
no-op shims into the env's site-packages. `fcntl` is provided in `tools/windows_shims/fcntl.py`;
install it with:
```powershell
$sp = python -c "import site; print(site.getsitepackages()[-1])"
Copy-Item .\tools\windows_shims\fcntl.py (Join-Path $sp 'fcntl.py')
```
If later imports fail on `resource`, `pwd`, or `grp`, tell me and I'll add matching shims.

> **Platform caveat (important for scope):** these shims are enough for the **nuScenes
> open-loop** metrics (L2 + collision), which never run nuplan simulation. The **nuPlan
> closed-loop PDMS** path (the paper's headline metric) needs real nuplan-devkit map/sim code
> and effectively requires **Linux or WSL2** — it will not run natively on Windows. Plan to move
> to WSL2 if/when you scale beyond nuScenes.

---

## GATE 4 — Download the pretrained model (~7 GB)
```powershell
# from qwen_architecture\AutoVLA
python tools\download\download_qwen.py --repo_id Qwen/Qwen2.5-VL-3B-Instruct --local_dir .\Qwen2.5-VL-3B-Instruct
dir .\Qwen2.5-VL-3B-Instruct   # expect config.json, model-*.safetensors, tokenizer files
```
**Pass if:** the safetensors shards are present.

---

## GATE 5 — Preprocess nuScenes-mini (train + val splits)
This needs `nuscenes-devkit`, which conflicts with the main stack — use a **separate** env:
```powershell
conda create -n nusc_preprocess python=3.9 -y
conda activate nusc_preprocess
pip install nuscenes-devkit==1.1.11 pyquaternion numpy tqdm

# from qwen_architecture\AutoVLA  (run each split; --version v1.0-mini is the key flag)
python tools\preprocessing\nusc_sample_generation.py --nuscenes_path .\dataset\nuscenes --output_dir .\dataset\nuscenes\nuscenes_train --split train --version v1.0-mini
python tools\preprocessing\nusc_sample_generation.py --nuscenes_path .\dataset\nuscenes --output_dir .\dataset\nuscenes\nuscenes_val   --split val   --version v1.0-mini

conda activate qwen_finetune
dir .\dataset\nuscenes\nuscenes_train\*.json   # expect several .json files (mini_train scenes)
dir .\dataset\nuscenes\nuscenes_val\*.json     # expect a few .json files (mini_val scenes)
```
**Pass if:** both folders contain `.json` scene files.
> Note: the config points at `./dataset/nuscenes/nuscenes_train` and `nuscenes_val` — matched here.
> (These per-scene JSONs store the image paths from preprocessing, which is why the config uses
> `sensor_data_path: null`.)

---

## GATE 6 — LoRA SFT smoke test (the real pipeline gate)
```powershell
# from qwen_architecture\AutoVLA
python tools\run_sft_lora.py --config training/qwen2.5-vl-3B-nusc-sft-lora --train_sample_size 8 --epochs 1 --limit_val_batches 2
```
Watch, in another terminal: `nvidia-smi` (VRAM should be well under 32 GB).
**Pass if:** you see `print_trainable_parameters()` (LoRA + embed/lm_head only, a few % of 3B),
the loss prints and decreases, and a checkpoint lands in `runs\sft_lora\<timestamp>\`.
**Paste back:** the `trainable params` line, peak VRAM, and any traceback.

Full mini run once the smoke passes:
```powershell
python tools\run_sft_lora.py --config training/qwen2.5-vl-3B-nusc-sft-lora
```

---

## GATE 7 — Merge adapter → eval-ready checkpoint
```powershell
python tools\merge_lora.py --config config\training\qwen2.5-vl-3B-nusc-sft-lora.yaml --adapter_ckpt runs\sft_lora\<timestamp>\epoch=..-loss=...ckpt --out checkpoints\nusc_sft_lora_merged.ckpt
```

---

## GATE 8 — nuScenes eval (L2 + collision)
Download the UniAD-style segmentation `.pt` files first (link in README → nuScenes Evaluation),
put them somewhere like `.\dataset\nusc_eval_seg\`, then:
```powershell
python tools\eval\nusc_eval.py --config config\eval\qwen2.5-vl-3B-nusc-sft-eval.yaml --checkpoint checkpoints\nusc_sft_lora_merged.ckpt --seg_data_path .\dataset\nusc_eval_seg --output outputs\planning_table_mini.txt
```
**Pass if:** it prints the L2 + collision tables and writes `outputs\planning_table_mini.txt`.

---

## Scale-up (after mini is green end-to-end)
1. Download nuScenes `v1.0-trainval`; put it under `datasets\nuscenes` (same junction).
2. Re-run GATE 5 with `--version v1.0-trainval`.
3. `python tools\run_sft_lora.py --config training/qwen2.5-vl-3B-nusc-sft-lora`  (full split).
4. Re-merge (GATE 7) and re-eval (GATE 8).
5. Sanity vs released checkpoint `Zewei-Zhou/AutoVLA` before trusting your own numbers.
