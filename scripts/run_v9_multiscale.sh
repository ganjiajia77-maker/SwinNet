#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
PY=${PYTHON:-/home/gjj/miniconda3/envs/swinunet/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-2}
export OPENCV_LOG_LEVEL=ERROR
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN_DIR=${RUN_DIR:-/home/gjj/SwinNet-roadbias/model_out/h0_v9_direct_20261028_135407}
SOURCE=${SOURCE_CHECKPOINT:-$RUN_DIR/stage_D/best.pth}
DATA=${DATA:-/home/gjj/Swin-Unet-main/Massachusetts/tiff}
COMMAND=${1:-smoke}
if [[ -z ${MS_ROOT:-} ]]; then
  if [[ -s "$RUN_DIR/latest_multiscale_root.txt" ]]; then
    MS_ROOT=$(cat "$RUN_DIR/latest_multiscale_root.txt")
  elif [[ "$COMMAND" == smoke || "$COMMAND" == all ]]; then
    MS_ROOT="$RUN_DIR/multiscale_$(date +%Y%m%d_%H%M%S)"
  else
    echo 'Run smoke first, or set MS_ROOT to the existing paired experiment directory.' >&2
    exit 1
  fi
fi
[[ -x "$PY" && -s "$SOURCE" ]] || {
  echo "Missing Python or original V9 checkpoint: $PY / $SOURCE" >&2; exit 1; }
mkdir -p "$MS_ROOT"
printf '%s\n' "$MS_ROOT" > "$RUN_DIR/latest_multiscale_root.txt"
common=(--source_checkpoint "$SOURCE" --root_path "$DATA" --output_dir "$MS_ROOT"
  --device cuda --seed "${SEED:-1234}" --batch_size 2 --num_workers "${WORKERS:-4}"
  --epochs "${EPOCHS:-20}" --val_interval "${VAL_INTERVAL:-5}" --transition_epochs "${TRANSITION_EPOCHS:-3}")
if [[ -n ${SELECTOR_LR:-} ]]; then common+=(--selector_lr "$SELECTOR_LR"); fi
echo "SOURCE=$SOURCE"
echo "MS_ROOT=$MS_ROOT; GPU=${GPU:-2}; extra epochs per arm=${EPOCHS:-20}; loss=L_seg+0.2*L_cont"
run() { "$PY" -u run_v9_multiscale.py "$1" "${common[@]}" "${@:2}"; }
case "$COMMAND" in
  smoke) run smoke ;;
  train_local|train_multiscale|resume_local|resume_multiscale|sweep_local|sweep_multiscale|test_local|test_multiscale)
    action=${COMMAND%%_*}; mode=${COMMAND#*_}
    run "$action" --transport_mode "$mode" ;;
  train_pair) run train --transport_mode local; run train --transport_mode multiscale ;;
  sweep) run sweep --transport_mode local; run sweep --transport_mode multiscale ;;
  test) run test --transport_mode local; run test --transport_mode multiscale ;;
  audit|audit_test)
    split=val; [[ "$COMMAND" == audit_test ]] && split=test
    extra=(); [[ -n ${REFERENCE_SET:-} ]] && extra+=(--reference_set "$REFERENCE_SET")
    run audit --transport_mode multiscale --audit_split "$split" "${extra[@]}"
    ;;
  all)
    run smoke
    run train --transport_mode local; run train --transport_mode multiscale
    run sweep --transport_mode local; run sweep --transport_mode multiscale
    run audit --transport_mode multiscale --audit_split val
    ;;
  *) echo "Unknown command: $COMMAND" >&2; exit 1 ;;
esac
