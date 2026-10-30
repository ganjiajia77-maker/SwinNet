#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export PYTHONUNBUFFERED=1
RESULTS=${RESULTS:-/home/gjj/SwinNet-roadbias/model_out}
RUN_DIR=${RUN_DIR:-$RESULTS/h0_v9_direct_20261028_135407}
if [[ -z ${GATE_BENEFIT_RUN:-} ]]; then
  [[ -s "$RESULTS/latest_v9_gate_benefit_run.txt" ]] || { echo 'Missing latest_v9_gate_benefit_run.txt; set GATE_BENEFIT_RUN to the completed paired experiment.' >&2; exit 1; }
  GATE_BENEFIT_RUN=$(cat "$RESULTS/latest_v9_gate_benefit_run.txt")
fi
FIXED_ACTION_OUT=${FIXED_ACTION_OUT:-$GATE_BENEFIT_RUN/fixed_action_equal_helpful_v1}
MODE=${1:-run}
if [[ "$MODE" == results ]]; then
  [[ -s "$FIXED_ACTION_OUT/REPORT.md" ]] || { echo "Report not yet available: $FIXED_ACTION_OUT/REPORT.md" >&2; exit 1; }
  cat "$FIXED_ACTION_OUT/REPORT.md"
  echo "CSV=$FIXED_ACTION_OUT/matched_helpful.csv"
  exit 0
fi
mkdir -p "$FIXED_ACTION_OUT"
exec 9>"$FIXED_ACTION_OUT/.run.lock"
flock -n 9 || { echo "Diagnostic already running: $FIXED_ACTION_OUT" >&2; exit 1; }
common=(--run_dir "$RUN_DIR" --experiment_dir "$GATE_BENEFIT_RUN"
  --root_path "${DATA:-/home/gjj/Swin-Unet-main/Massachusetts/tiff}" --output_dir "$FIXED_ACTION_OUT"
  --device cuda --bootstrap "${BOOTSTRAP:-100}")
case "$MODE" in
  run)
    python -m unittest checks.test_v9_fixed_actions 2>&1 | tee -a "$FIXED_ACTION_OUT/unit_tests.log"
    python -u diagnose_v9_fixed_actions.py "${common[@]}" 2>&1 | tee -a "$FIXED_ACTION_OUT/console.log"
    ;;
  cached)
    python -u diagnose_v9_fixed_actions.py "${common[@]}" --cached_only 2>&1 | tee -a "$FIXED_ACTION_OUT/console.log"
    ;;
  *) echo 'Usage: run_v9_fixed_actions.sh run|cached|results' >&2; exit 2 ;;
esac
