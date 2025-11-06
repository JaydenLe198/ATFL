# ATFL (Anomaly Transformer Federated Learning)

Scaffold for a Flower + PyTorch simulation that federates the original
[Anomaly-Transformer](https://github.com/huyln15/Anomaly-Transformer) model.
The goal is to train the time-domain transformer across simulated clients with
FedAvg, FedProx, or SCAFFOLD while producing the same anomaly-detection
artifacts as the centralized project.

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

## Next Steps

1. Port the original `AnomalyTransformer` implementation into `fl/model/TimeTransformer.py`
   (or reuse via import if keeping the repo as a submodule).
2. Fill in dataset loaders in `fl/utils/data_loader.py` for PSM, SMD, SMAP, and MSL.
3. Implement the actual training loop logic in `fl/task.py`, mirroring the centralized
   project while adding FedProx and SCAFFOLD hooks.
4. Ready-to-run YAML configs live in `fl/configs/` (e.g., `psm_fedprox.yaml`, `smd_scaffold.yaml`, etc.).
5. Drop raw datasets under `fl/datasets/` using the same structure as the original project.
6. Optional: enable Weights & Biases by setting `wandb.enabled: true` in your YAML and exporting `WANDB_API_KEY`.

See `fl/run_simulation.py` for the intended workflow. For batch experiments, run `bash fl/run_all_baselines.sh` from the project root.
