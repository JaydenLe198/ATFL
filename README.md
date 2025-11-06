# ATFL (Anomaly Transformer Federated Learning)

Federated Flower + PyTorch replica of the original
[Anomaly-Transformer](https://github.com/huyln15/Anomaly-Transformer) project.
It trains the time-domain transformer across simulated clients using FedAvg,
FedProx, or SCAFFOLD and evaluates with the same TranAD-style metrics as the
centralized baseline.

## Layout

- `fl/` &mdash; Flower application package  
  - `client_app.py` &mdash; Flower `NumPyClient` wrapper around Anomaly Transformer  
  - `server_app.py` &mdash; optional Flower `ServerApp` for `flwr run`  
  - `task.py` &mdash; model loading, training, and metric utilities  
  - `run_simulation.py` &mdash; Ray/Flower simulator entry point (runs TranAD-style evaluation after training)  
  - `evaluation_cpu.py` &mdash; optional offline evaluation for saved time arrays (legacy support)  
  - `utils/` &mdash; config loader, dataset tooling, caching, evaluation helpers, wandb utilities  
  - `configs/` &mdash; YAML presets (default + dataset-specific)  
  - `model/` &mdash; wrapper for the original Anomaly Transformer modules  
  - output folders (`eval_results/`, `logs/`, `out/`) for artefacts and metrics
- `pyproject.toml` &mdash; Flower app metadata and dependencies  
- `instruction.md` &mdash; step-by-step setup guide

## Requirements

- Python ≥ 3.9
- CUDA-capable GPU (optional but recommended)
- Datasets from the original Anomaly-Transformer repository (PSM, SMD, SMAP, MSL)

## Setup

```bash
cd /home/nhle4237/ATFL           # adjust to your clone path
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Place datasets under `fl/datasets/` using the same structure as in the original
project, e.g.:

```
fl/datasets/PSM/train.csv
fl/datasets/SMD/train/<machine>.csv
fl/datasets/SMAP+MSL/data/train/<channel>.csv
```

## Running a smoke test

```bash
source .venv/bin/activate
python3 -m fl.run_simulation fl/configs/psm_smoke.yaml
```

- Uses 2 clients, 1 training round, and GPU resources (if available).
- Reports device assignment (`[Client X] using GPU ...`) and logs TranAD metrics
  to `eval_results/smoke/PSM_tranad_metrics.csv`.

## Full baseline scripts

Each dataset × strategy pair has a YAML preset in `fl/configs/`:

| Dataset | FedProx config              | SCAFFOLD config                 |
|---------|----------------------------|---------------------------------|
| PSM     | `psm_fedprox.yaml`         | `psm_scaffold.yaml`             |
| SMD     | `smd_fedprox.yaml`         | `smd_scaffold.yaml`             |
| SMAP    | `smap_fedprox.yaml`        | `smap_scaffold.yaml`            |
| MSL     | `msl_fedprox.yaml`         | `msl_scaffold.yaml`             |

Run everything sequentially (similar to DualTF-FLsim):

```bash
bash fl/run_all_baselines.sh
```

Outputs per run:
- Terminal log and W&B metrics streamed to `logs/<timestamp>/...`
- Config, run log, and TranAD CSV copied to `out/<dataset>/<strategy>/<timestamp>/`
- Raw metrics stored in `eval_results/<timestamp>_<dataset>_<strategy>/...`

## Weights & Biases

All presets have W&B enabled with:
```yaml
wandb:
  enabled: true
  project: ATFL
  entity: huyln
  run_name: <dataset>_<strategy>_30r
  tags: [...]
```

Before running:
```bash
export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxx
# optional: export WANDB_MODE=offline
```

Logged values:
- Per-round `clients/time_mean_s`, `clients/time_std_s`, and per-client losses.
- Post-training `eval/auc`, `eval/f1`, `eval/f1_adjusted`, etc.
- Centralized inference time `post/infer_time_s`.

## Dataset partitioning & model settings

- PSM uses sequential partitioning (24 clients).
- SMD, SMAP, MSL use by-machine partitioning (matching available segments).
- Sequence length and feature dimensions mirror the original AT settings.
- Batch sizes default to 64; adjust `batch_size` in the YAMLs as needed.

## Troubleshooting

- **No GPU usage**: ensure `resources.client.num_gpus` in your YAML is non-zero
  and run `nvidia-smi` during training.
- **Missing datasets**: verify paths under `fl/datasets/` and run
  `python3 -m fl.utils.check_datasets --dataset <name>` for quick validation.
- **W&B not logging**: confirm `wandb.enabled: true`, API key export, and that
  the `wandb` package is installed in the virtual environment.
- **Push to GitHub**: `.gitignore` excludes datasets, logs, and cache folders;
  only source/config/docs are tracked.
