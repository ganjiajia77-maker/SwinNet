#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export PYTHONUNBUFFERED=1
DATA=${DATA:-/home/gjj/Swin-Unet-main/Massachusetts/tiff}
RUN_DIR=${RUN_DIR:-/home/gjj/SwinNet-roadbias/model_out/h0_v9_direct_20261028_135407}
MODE=${1:-oracle}
[[ -s "$RUN_DIR/stage_D/best.pth" && -s "$RUN_DIR/direct_val_selection.json" ]] || {
  echo 'Missing V9 best.pth or direct_val_selection.json' >&2; exit 1; }
if [[ -z ${V92_ROOT:-} ]]; then
  if [[ -s "$RUN_DIR/latest_v92_root.txt" ]]; then V92_ROOT=$(cat "$RUN_DIR/latest_v92_root.txt")
  else V92_ROOT="$RUN_DIR/v92_$(date +%Y%m%d_%H%M%S)"; fi
fi
mkdir -p "$V92_ROOT"
printf '%s\n' "$V92_ROOT" > "$RUN_DIR/latest_v92_root.txt"
output="$V92_ROOT/${MODE}_$(date +%Y%m%d_%H%M%S_%N)"
echo "RUN_DIR=$RUN_DIR"
echo "V92_ROOT=$V92_ROOT"
echo "OUTPUT=$output"
case "$MODE" in
  oracle)
    if [[ -z ${REFERENCE_SET:-} ]]; then
      [[ -s "$RUN_DIR/latest_v91_root.txt" ]] || { echo 'Missing V9.1 root pointer' >&2; exit 1; }
      v91root=$(cat "$RUN_DIR/latest_v91_root.txt")
      [[ -s "$v91root/latest_counterfactual.txt" ]] || { echo 'Missing V9.1 counterfactual pointer' >&2; exit 1; }
      cf=$(cat "$v91root/latest_counterfactual.txt")
      REFERENCE_SET="$cf/reference_set.json"
    fi
    [[ -s "$REFERENCE_SET" ]] || { echo "Missing fixed V9.1 reference set: $REFERENCE_SET" >&2; exit 1; }
    python -u diagnostics/v92_oracle_gate.py --run_dir "$RUN_DIR" --root_path "$DATA" \
      --reference_set "$REFERENCE_SET" --output_dir "$output" --device cuda
    ;;
  train_paths)
    python -u diagnostics/v92_train_audit.py --run_dir "$RUN_DIR" --root_path "$DATA" \
      --output_dir "$output" --device cuda --thresholds .30 .45 .60
    ;;
  train_smoke)
    python -u diagnostics/v92_train_audit.py --run_dir "$RUN_DIR" --root_path "$DATA" \
      --output_dir "$output" --device cuda --thresholds .30 .45 .60 --max_images 1
    ;;
  *) echo 'Usage: run_v92_mass.sh oracle | train_smoke | train_paths' >&2; exit 2 ;;
esac
printf '%s\n' "$output" > "$V92_ROOT/latest_${MODE}.txt"
echo "DONE=$output"
