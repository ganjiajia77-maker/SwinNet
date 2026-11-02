#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
MODE=${1:-all}
case "$MODE" in all|smoke|train|val|compare) ;; *) echo "Choose all|smoke|train|val|compare"; exit 2;; esac
PAIR_DIR=${PAIR_DIR:-/home/gjj/SwinNet-roadbias/model_out/full_h0_joint_pair_20e_20261101_085322}
SOURCE_RUN=${SOURCE_RUN:-B_transport_0p01_proposal_0p001}
RUN_NAME=${RUN_NAME:-B_unscaled_same_initial_20e}
GPU=${GPU:-1}
[[ "$RUN_NAME" =~ ^B_[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid output run name'; exit 2; }
[[ "$SOURCE_RUN" =~ ^B_[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid source run name'; exit 2; }
[[ "$RUN_NAME" != "$SOURCE_RUN" ]] || { echo 'Output must differ from source'; exit 2; }
mkdir -p "$PAIR_DIR/$RUN_NAME"
exec 9>"$PAIR_DIR/$RUN_NAME/.run.lock"
flock -n 9 || { echo "This run is already active"; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU"
python -u run_full_h0_proposal_adapt.py "$MODE" --pair_dir "$PAIR_DIR" \
  --source_run "$SOURCE_RUN" --run_name "$RUN_NAME" --epochs 20 --device cuda \
  2>&1 | tee -a "$PAIR_DIR/$RUN_NAME/console.log"
