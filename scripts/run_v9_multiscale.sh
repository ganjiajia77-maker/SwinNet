#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
PY=${PYTHON:-/home/gjj/miniconda3/envs/swinunet/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-2}
export OPENCV_LOG_LEVEL=ERROR
export PYTHONUNBUFFERED=1
# The server has an older driver. Do not force CUDA virtual-memory mappings.
# Set both aliases, since newer torch prefers PYTORCH_ALLOC_CONF.
export PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:False
export PYTORCH_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
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
# UUID selection avoids ambiguity between NVML physical indices and CUDA's
# default device ordering. Check before each phase; never kill occupying jobs.
gpu_preflight() {
  local uuid
  uuid=$("$PY" scripts/v9_gpu_preflight.py --gpu "${GPU:-2}" \
    --min_free_mib "${MIN_FREE_GPU_MIB:-6144}")
  export CUDA_VISIBLE_DEVICES="$uuid"
}
run() {
  gpu_preflight
  if [[ "$1" == smoke ]]; then
    # Synchronize only the preflight so any further CUDA failure reports the
    # actual operation and C++ stack. Formal training keeps asynchronous CUDA.
    CUDA_LAUNCH_BLOCKING=1 TORCH_SHOW_CPP_STACKTRACES=1 \
      "$PY" -u run_v9_multiscale.py "$1" "${common[@]}" "${@:2}"
  else
    "$PY" -u run_v9_multiscale.py "$1" "${common[@]}" "${@:2}"
  fi
}
continue_arm() {
  local mode=$1 directory="$MS_ROOT/$1"
  if [[ -s "$directory/last.pth" ]]; then
    echo "[CONTINUE] $mode from $directory/last.pth"
    run resume --transport_mode "$mode"
  else
    if [[ -d "$directory" ]]; then
      # A first-epoch OOM leaves experiment.json but no resumable weights.
      # Preserve every file; do not silently discard a lone best/temporary save.
      if [[ -n $(find "$directory" -type f \( -name '*.pth' -o -name '*.tmp' \) -print -quit) ]]; then
        echo "Missing last.pth but checkpoint files exist in $directory; inspect before continuing." >&2
        return 1
      fi
      local backup="${directory}_interrupted_no_checkpoint_$(date +%Y%m%d_%H%M%S_%N)"
      mv -- "$directory" "$backup"
      echo "[PRESERVED] incomplete $mode directory moved to $backup"
    fi
    run train --transport_mode "$mode"
  fi
}
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
  continue_all)
    run smoke
    continue_arm local; continue_arm multiscale
    run sweep --transport_mode local; run sweep --transport_mode multiscale
    run audit --transport_mode multiscale --audit_split val
    run test --transport_mode local; run test --transport_mode multiscale
    run audit --transport_mode multiscale --audit_split test
    echo '[ALL COMPLETE] paired continuations, validation selection, test and audits'
    ;;
  *) echo "Unknown command: $COMMAND" >&2; exit 1 ;;
esac
