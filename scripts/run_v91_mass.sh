#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export PYTHONUNBUFFERED=1
DATA=${DATA:-/home/gjj/Swin-Unet-main/Massachusetts/tiff}
RUN_DIR=${RUN_DIR:-/home/gjj/SwinNet-roadbias/model_out/h0_v9_direct_20261028_135407}
MODE=${1:-counterfactual}
if [[ -z ${V91_ROOT:-} ]]; then
  if [[ -s "$RUN_DIR/latest_v91_root.txt" ]]; then
    V91_ROOT=$(cat "$RUN_DIR/latest_v91_root.txt")
  else
    V91_ROOT="$RUN_DIR/v91_$(date +%Y%m%d_%H%M%S)"
  fi
fi
mkdir -p "$V91_ROOT"
printf '%s\n' "$V91_ROOT" > "$RUN_DIR/latest_v91_root.txt"
[[ -s "$RUN_DIR/stage_D/best.pth" && -s "$RUN_DIR/direct_val_selection.json" ]] || {
  echo 'Missing V9 best.pth or val selection. Run the V9 val_direct command first.' >&2; exit 1; }
OUTPUT_THRESHOLD=$(python - "$RUN_DIR/direct_val_selection.json" <<'PY'
import json, sys
item = json.load(open(sys.argv[1]))
if item.get('selected_on') != 'val':
    raise SystemExit('A validation-selected threshold is required')
print(item['threshold'])
PY
)
echo "RUN_DIR=$RUN_DIR"
echo "V91_ROOT=$V91_ROOT"
echo "val_threshold=$OUTPUT_THRESHOLD; GPU=${GPU:-1}"

new_output() {
  printf '%s/%s_%s\n' "$V91_ROOT" "$1" "$(date +%Y%m%d_%H%M%S_%N)"
}
save_latest() {
  local kind=$1 path=$2 tmp="$V91_ROOT/.latest_$1.tmp"
  printf '%s\n' "$path" > "$tmp"
  mv -f "$tmp" "$V91_ROOT/latest_$kind.txt"
}
read_latest() {
  local kind=$1 marker="$V91_ROOT/latest_$1.txt"
  [[ -s "$marker" ]] || { echo "Missing successful $kind output pointer: $marker" >&2; exit 1; }
  local path
  path=$(cat "$marker")
  [[ -d "$path" ]] || { echo "Recorded $kind output directory is missing: $path" >&2; exit 1; }
  printf '%s\n' "$path"
}

case "$MODE" in
  counterfactual)
    output=$(new_output counterfactual)
    echo "OUTPUT=$output"
    python -u diagnostics/v91_counterfactual.py --run_dir "$RUN_DIR" --root_path "$DATA" \
      --output_dir "$output" --device cuda --expected_reference_gaps "${EXPECTED_GAPS:-206}"
    save_latest counterfactual "$output"
    ;;
  candidates)
    counterfactual=$(read_latest counterfactual)
    output=$(new_output train_candidates)
    echo "REFERENCE_SET=$counterfactual/reference_set.json"
    echo "OUTPUT=$output"
    python -u run_v91_gate_experiment.py candidates --run_dir "$RUN_DIR" --root_path "$DATA" \
      --output_dir "$output" --device cuda --batch_size 2 \
      --num_workers "${WORKERS:-4}" --seed "${SEED:-1234}" --output_threshold "$OUTPUT_THRESHOLD"
    save_latest candidates "$output"
    ;;
  train_control|train_gate|resume_control|resume_gate)
    counterfactual=$(read_latest counterfactual)
    candidates=$(read_latest candidates)
    [[ -s "$counterfactual/counterfactual.json" && -s "$candidates/candidates_train.json" ]] || {
      echo 'Run counterfactual and the full train-set candidate audit first.' >&2; exit 1; }
    variant=${MODE#*_}
    command=${MODE%%_*}
    python -u run_v91_gate_experiment.py "$command" --run_dir "$RUN_DIR" --root_path "$DATA" \
      --output_dir "$V91_ROOT/continuations" --candidate_report "$candidates/candidates_train.json" \
      --variant "$variant" --device cuda --epochs "${EPOCHS:-20}" --val_interval 5 --batch_size 2 \
      --num_workers "${WORKERS:-4}" --seed "${SEED:-1234}" --lambda_gate "${LAMBDA_GATE:-0.1}" \
      --output_threshold "$OUTPUT_THRESHOLD"
    ;;
  compare)
    counterfactual=$(read_latest counterfactual)
    output=$(new_output compare)
    echo "REFERENCE_SET=$counterfactual/reference_set.json"
    echo "OUTPUT=$output"
    python -u diagnostics/v91_compare_gate.py --run_dir "$RUN_DIR" --root_path "$DATA" \
      --experiment_dir "$V91_ROOT/continuations" --output_dir "$output" \
      --reference_set "$counterfactual/reference_set.json" --device cuda \
      --expected_reference_gaps "${EXPECTED_GAPS:-206}"
    save_latest compare "$output"
    ;;
  *) echo 'Usage: counterfactual | candidates | train_control | train_gate | resume_control | resume_gate | compare' >&2; exit 2 ;;
esac
echo "V91_ROOT=$V91_ROOT"
