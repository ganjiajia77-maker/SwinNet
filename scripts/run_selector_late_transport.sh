#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export PYTHONUNBUFFERED=1
SELECTOR_RUN=${SELECTOR_RUN:-/home/gjj/SwinNet-roadbias/model_out/h0_v9_source_selector_20261030_110206}
if [[ -z ${TRANSPORT_RUN:-} ]]; then
  [[ -s "$SELECTOR_RUN/latest_actual_transport_run.txt" ]] || { echo 'Missing actual transport run marker; set TRANSPORT_RUN.' >&2; exit 1; }
  TRANSPORT_RUN=$(cat "$SELECTOR_RUN/latest_actual_transport_run.txt")
fi
DIAG_OUT=${DIAG_OUT:-$TRANSPORT_RUN/late_transport_contrasts_v1}
MODE=${1:-run}
if [[ "$MODE" == results ]]; then
  [[ -s "$DIAG_OUT/REPORT.md" ]] || { echo "Report not yet available: $DIAG_OUT/REPORT.md" >&2; exit 1; }
  cat "$DIAG_OUT/REPORT.md"
  echo "STEP_CSV=$DIAG_OUT/step_comparison.csv"
  echo "PER_IMAGE_CSV=$DIAG_OUT/per_image_steps.csv"
  exit 0
fi
[[ "$MODE" == run ]] || { echo 'Usage: run_selector_late_transport.sh run|results' >&2; exit 2; }
mkdir -p "$DIAG_OUT"
exec 9>"$DIAG_OUT/.run.lock"
flock -n 9 || { echo "Already running: $DIAG_OUT" >&2; exit 1; }
args=(--run_dir "$TRANSPORT_RUN" --output_dir "$DIAG_OUT" --device cuda --trace_images "${TRACE_IMAGES:-4}")
if [[ -n ${THRESHOLD:-} ]]; then args+=(--threshold "$THRESHOLD"); fi
if [[ -n ${CHECKPOINT:-} ]]; then args+=(--checkpoint "$CHECKPOINT"); fi
python -u diagnose_selector_late_transport.py "${args[@]}" 2>&1 | tee -a "$DIAG_OUT/console.log"
