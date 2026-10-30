#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export PYTHONUNBUFFERED=1
RESULTS=${RESULTS:-/home/gjj/SwinNet-roadbias/model_out}
SELECTOR_RUN=${SELECTOR_RUN:-$RESULTS/h0_v9_source_selector_20261030_110206}
MODE=${1:-all}
if [[ -z ${TRANSPORT_RUN:-} ]]; then
  if [[ -s "$SELECTOR_RUN/latest_actual_transport_run.txt" ]]; then TRANSPORT_RUN=$(cat "$SELECTOR_RUN/latest_actual_transport_run.txt")
  else TRANSPORT_RUN="$SELECTOR_RUN/actual_transport_$(date +%Y%m%d_%H%M%S)"; fi
fi
mkdir -p "$TRANSPORT_RUN"
printf '%s\n' "$TRANSPORT_RUN" > "$SELECTOR_RUN/latest_actual_transport_run.txt"
if [[ "$MODE" == results ]]; then
  for file in preflight.json actual_update_check.json REPORT.md fixed_relation_after/REPORT.md; do
    if [[ -s "$TRANSPORT_RUN/$file" ]]; then cat "$TRANSPORT_RUN/$file"; fi
  done
  echo "TRANSPORT_RUN=$TRANSPORT_RUN"
  exit 0
fi
exec 9>"$TRANSPORT_RUN/.run.lock"
flock -n 9 || { echo "Already running: $TRANSPORT_RUN" >&2; exit 1; }
args=(--selector_run "$SELECTOR_RUN" --output_dir "$TRANSPORT_RUN" --results "$RESULTS"
  --device cuda --epochs "${EPOCHS:-20}" --bootstrap "${BOOTSTRAP:-100}")
if [[ -n ${CONTROL_RUN:-} ]]; then args+=(--control_run "$CONTROL_RUN"); fi
if [[ -n ${ORIGINAL_RUN:-} ]]; then args+=(--original_run "$ORIGINAL_RUN"); fi
if [[ -n ${REFERENCE_SET:-} ]]; then args+=(--reference_set "$REFERENCE_SET"); fi
if [[ -n ${LAMBDA_RELATION:-} ]]; then args+=(--lambda_relation "$LAMBDA_RELATION"); fi
case "$MODE" in
  all|preflight|smoke|train|val)
    python -u run_v9_selector_transport.py "$MODE" "${args[@]}" 2>&1 | tee -a "$TRANSPORT_RUN/console.log"
    ;;
  *) echo 'Usage: run_v9_selector_transport.sh all|preflight|smoke|train|val|results' >&2; exit 2 ;;
esac
