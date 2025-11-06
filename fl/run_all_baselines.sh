#!/usr/bin/env bash

# Batch runner for ATFL experiments.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUNSTAMP=$(date +"%Y%m%d_%H%M%S")
LOGDIR="logs/${RUNSTAMP}"
mkdir -p "$LOGDIR"

RUNS=(
  "PSM FEDPROX fl/configs/psm_fedprox.yaml"
  "PSM SCAFFOLD fl/configs/psm_scaffold.yaml"
  "SMD FEDPROX fl/configs/smd_fedprox.yaml"
  "SMD SCAFFOLD fl/configs/smd_scaffold.yaml"
  "SMAP FEDPROX fl/configs/smap_fedprox.yaml"
  "SMAP SCAFFOLD fl/configs/smap_scaffold.yaml"
  "MSL FEDPROX fl/configs/msl_fedprox.yaml"
  "MSL SCAFFOLD fl/configs/msl_scaffold.yaml"
)

run_one() {
  local DATASET="$1"; shift
  local STRAT="$1"; shift
  local CFG="$1"; shift

  local RUNID="${DATASET}_${STRAT}"
  local LOGFILE="${LOGDIR}/${RUNID}.log"
  local OUTDIR="out/${DATASET}/${STRAT}/${RUNSTAMP}"
  local EVALDIR="eval_results/${RUNSTAMP}_${RUNID}"
  mkdir -p "$OUTDIR"

  echo "==== [${RUNID}] START $(date -Is) ====" | tee -a "$LOGFILE"
  echo "Config: $CFG" | tee -a "$LOGFILE"
  echo "Eval dir: $EVALDIR" | tee -a "$LOGFILE"

  if ! python3 -m fl.run_simulation "$CFG" --results-dir "$EVALDIR" |& tee -a "$LOGFILE"; then
    echo "[${RUNID}] Training failed (see log)." | tee -a "$LOGFILE"
  else
    echo "[${RUNID}] Training completed." | tee -a "$LOGFILE"
  fi

  local METRIC_SRC="${EVALDIR}/${DATASET}_tranad_metrics.csv"
  if [[ -f "$METRIC_SRC" ]]; then
    cp -v "$METRIC_SRC" "$OUTDIR/${RUNID}_tranad_metrics.csv" | tee -a "$LOGFILE"
  else
    echo "[${RUNID}] Metrics file not found at $METRIC_SRC" | tee -a "$LOGFILE"
  fi

  cp -v "$CFG" "$OUTDIR/config_used.yaml" | tee -a "$LOGFILE" || true
  cp -v "$LOGFILE" "$OUTDIR/run.log" | tee -a "$LOGFILE" || true

  cat >"$OUTDIR/manifest.json" <<EOF
{
  "dataset": "${DATASET}",
  "strategy": "${STRAT}",
  "config": "${CFG}",
  "log": "${LOGFILE}",
  "eval_results": "${EVALDIR}",
  "timestamp": "${RUNSTAMP}"
}
EOF

  echo "==== [${RUNID}] DONE $(date -Is) ====" | tee -a "$LOGFILE"
}

for spec in "${RUNS[@]}"; do
  # shellcheck disable=SC2086
  run_one $spec
  echo
done

echo "Logs written to $LOGDIR"
