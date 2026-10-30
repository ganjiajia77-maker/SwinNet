#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export PYTHONUNBUFFERED=1
SELECTOR_RUN=${SELECTOR_RUN:-/home/gjj/SwinNet-roadbias/model_out/h0_v9_source_selector_20261030_110206}
if [[ -z ${TRANSPORT_RUN:-} ]]; then
  [[ -s "$SELECTOR_RUN/latest_actual_transport_run.txt" ]] || { echo 'Missing current S_selector marker; set TRANSPORT_RUN.' >&2; exit 1; }
  TRANSPORT_RUN=$(cat "$SELECTOR_RUN/latest_actual_transport_run.txt")
fi
MODE=${1:-all}
if [[ -z ${CHOICE_RUN:-} ]]; then
  if [[ -s "$TRANSPORT_RUN/latest_choice_loss_run.txt" ]]; then
    CHOICE_RUN=$(cat "$TRANSPORT_RUN/latest_choice_loss_run.txt")
  else
    CHOICE_RUN="$TRANSPORT_RUN/late_choice_$(date +%Y%m%d_%H%M%S)"
  fi
fi
if [[ "$MODE" == results ]]; then
  [[ -f "$CHOICE_RUN/REPORT.md" ]] || { echo "Not complete; see $CHOICE_RUN/console.log"; exit 1; }
  cat "$CHOICE_RUN/REPORT.md"
  echo "ACTUAL_RELATIONS=$CHOICE_RUN/actual_source_relations.csv"
  echo "LATE_RETENTION=$CHOICE_RUN/late_road_retention.csv"
  exit 0
fi
[[ "$MODE" == all || "$MODE" == train || "$MODE" == val || "$MODE" == smoke ]] || { echo 'Usage: all|smoke|train|val|results' >&2; exit 2; }
[[ -s "$TRANSPORT_RUN/best.pth" ]] || { echo "Missing CURRENT S_selector: $TRANSPORT_RUN/best.pth" >&2; exit 1; }
mkdir -p "$CHOICE_RUN"
printf '%s\n' "$CHOICE_RUN" > "$TRANSPORT_RUN/latest_choice_loss_run.txt"
exec 9>"$CHOICE_RUN/.run.lock"
flock -n 9 || { echo "Already running: $CHOICE_RUN" >&2; exit 1; }
args=("$MODE" --run_dir "$TRANSPORT_RUN" --output_dir "$CHOICE_RUN" --epochs "${EPOCHS:-20}" --lambda_choice "${LAMBDA_CHOICE:-0.05}" --device cuda)
if [[ -n ${THRESHOLD:-} ]]; then args+=(--threshold "$THRESHOLD"); fi
python -u run_v9_late_choice.py "${args[@]}" 2>&1 | tee -a "$CHOICE_RUN/console.log"
