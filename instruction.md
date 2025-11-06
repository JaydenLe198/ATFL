# ATFL Setup & Smoke Test Guide

These steps assume a Linux environment with Python ≥ 3.9 installed.

## 1. Create a Python virtual environment

```bash
cd /home/nhle4237/ATFL
python3 -m venv .venv
source .venv/bin/activate
```

To reuse the environment later:

```bash
source /home/nhle4237/ATFL/.venv/bin/activate
```

## 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs all packages needed for Flower, Ray, PyTorch, and evaluation.

## 3. Place datasets

Copy the raw datasets from the original Anomaly-Transformer project into the expected locations:

```
ATFL/fl/datasets/PSM/...
ATFL/fl/datasets/SMD/...
ATFL/fl/datasets/SMAP+MSL/...
```

Ensure the folder structure matches the source repository (e.g., `PSM/train.csv`, `SMD/train/*.csv`, `SMAP+MSL/data/train/*.csv`, etc.).

## 4. Run the smoke test

Activate the virtual environment if not already active, then run:

```bash
cd ATFL
python3 -m fl.run_simulation fl/configs/psm_smoke.yaml
```

This launches a minimal Flower simulation (2 clients, 1 round) and, on completion, computes TranAD-style metrics on the centralized test set. Results are written to `eval_results/smoke/PSM_tranad_metrics.csv`.

## 5. Batch runs

To replay multiple dataset/strategy combinations (similar to DualTF-FLsim), use:

```bash
cd ATFL
bash fl/run_all_baselines.sh
```

This script runs each experiment listed in `RUNS`, streams logs to both the terminal and `logs/<timestamp>/`, and copies the resulting TranAD metrics plus the config/log into `out/<dataset>/<strategy>/<timestamp>/`.

## 6. Enable Weights & Biases (optional)

1. Sign in to W&B and create an API key.
2. Export the key before running experiments:
   ```bash
   export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```
3. In your YAML config, set:
   ```yaml
   wandb:
     enabled: true
     project: ATFL
     entity: your_wandb_entity   # optional
     run_name: custom_name        # optional
   ```
   Adjust any other `wandb` fields (tags, notes, etc.) as needed.

During training, Flower rounds log per-client timing stats to W&B; after training, the centralized evaluation metrics (auc, f1, etc.) are logged as `eval/*`.

## 7. Next steps

- Adjust the config YAMLs under `fl/configs/` for larger runs (more clients, rounds, or different strategies).
- Enable dataset caching (`data.centralized_cache: true`) once you are satisfied with the setup to speed up repeated experiments.
