#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export PYTHONUNBUFFERED=1
RESULTS=${RESULTS:-/home/gjj/SwinNet-roadbias/model_out}
RUN_DIR=${RUN_DIR:-$RESULTS/h0_v9_direct_20261028_135407}
DATA=${DATA:-/home/gjj/Swin-Unet-main/Massachusetts/tiff}
MODE=${1:-all}
if [[ -z ${SOURCE_SELECTOR_RUN:-} ]]; then
  if [[ -s "$RESULTS/latest_v9_source_selector_run.txt" ]]; then SOURCE_SELECTOR_RUN=$(cat "$RESULTS/latest_v9_source_selector_run.txt")
  else SOURCE_SELECTOR_RUN="$RESULTS/h0_v9_source_selector_$(date +%Y%m%d_%H%M%S)"; fi
fi
mkdir -p "$SOURCE_SELECTOR_RUN"
printf '%s\n' "$SOURCE_SELECTOR_RUN" > "$RESULTS/latest_v9_source_selector_run.txt"
if [[ "$MODE" == results ]]; then
  for file in relation_best/REPORT.md pipeline_status.json repair_best/REPORT.md repair_surface_val/REPORT.md joint_surface_val/REPORT.md; do
    if [[ -s "$SOURCE_SELECTOR_RUN/$file" ]]; then cat "$SOURCE_SELECTOR_RUN/$file"; fi
  done
  echo "SOURCE_SELECTOR_RUN=$SOURCE_SELECTOR_RUN"
  exit 0
fi
exec 9>"$SOURCE_SELECTOR_RUN/.run.lock"
flock -n 9 || { echo "Another job already uses $SOURCE_SELECTOR_RUN" >&2; exit 1; }
[[ -s "$RUN_DIR/stage_D/best.pth" && -s "$RUN_DIR/direct_val_selection.json" ]] || {
  echo "Missing original V9 checkpoint/selection: $RUN_DIR" >&2; exit 1; }
common=(--run_dir "$RUN_DIR" --root_path "$DATA" --output_dir "$SOURCE_SELECTOR_RUN"
  --device cuda --relation_epochs "${RELATION_EPOCHS:-5}" --repair_epochs "${REPAIR_EPOCHS:-20}"
  --selector_lr "${SELECTOR_LR:-0.0001}" --joint_lr "${JOINT_LR:-0.000005}"
  --num_workers "${WORKERS:-4}" --seed "${SEED:-1234}" --bootstrap "${BOOTSTRAP:-100}")
echo "SOURCE_SELECTOR_RUN=$SOURCE_SELECTOR_RUN; GPU=${GPU:-1}; batch2 accumulation1 FP32 no EMA"
case "$MODE" in
  all)
    python -m unittest checks.test_v9_source_selector 2>&1 | tee -a "$SOURCE_SELECTOR_RUN/unit_tests.log"
    python -u run_v9_source_selector.py all "${common[@]}" 2>&1 | tee -a "$SOURCE_SELECTOR_RUN/console.log"
    ;;
  smoke|relation|audit|repair|joint|val)
    extra=()
    if [[ "$MODE" == audit ]]; then extra+=(--audit_phase "${AUDIT_PHASE:-relation}"); fi
    if [[ "$MODE" == val ]]; then extra+=(--phase "${PHASE:-repair}"); fi
    python -u run_v9_source_selector.py "$MODE" "${common[@]}" "${extra[@]}" 2>&1 | tee -a "$SOURCE_SELECTOR_RUN/console.log"
    ;;
  *) echo 'Usage: run_v9_source_selector.sh all|smoke|relation|audit|repair|joint|val|results' >&2; exit 2 ;;
esac
