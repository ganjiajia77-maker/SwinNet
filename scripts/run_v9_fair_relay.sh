#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
PY=${PYTHON:-/home/gjj/miniconda3/envs/swinunet/bin/python}
RUN_DIR=${RUN_DIR:-/home/gjj/SwinNet-roadbias/model_out/h0_v9_direct_20261028_135407}
MS_ROOT=${MS_ROOT:-$RUN_DIR/multiscale_fix_20261029_183251}
SOURCE=${SOURCE_CHECKPOINT:-$RUN_DIR/stage_D/best.pth}
DATA=${DATA:-/home/gjj/Swin-Unet-main/Massachusetts/tiff}
SPLIT=${1:-val}
[[ "$SPLIT" == val || "$SPLIT" == test ]] || { echo 'Use val or test.' >&2; exit 1; }
[[ -x "$PY" && -s "$SOURCE" && -s "$MS_ROOT/local/best.pth" && -s "$MS_ROOT/multiscale/best.pth" ]] || {
  echo 'Missing Python, source V9 checkpoint or one of the two trained best.pth checkpoints.' >&2; exit 1; }
export OPENCV_LOG_LEVEL=ERROR PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:False
export PYTORCH_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
CUDA_VISIBLE_DEVICES=$("$PY" scripts/v9_gpu_preflight.py --gpu "${GPU:-1}" --min_free_mib "${MIN_FREE_GPU_MIB:-6144}")
export CUDA_VISIBLE_DEVICES
OUTPUT="$MS_ROOT/fair_relay_${SPLIT}_$(date +%Y%m%d_%H%M%S_%N)"
echo "[OUTPUT] $OUTPUT"
"$PY" -u diagnostics/v9_multiscale_fair_and_relay.py --run "$MS_ROOT" --source "$SOURCE" \
  --data "$DATA" --output "$OUTPUT" --device cuda --split "$SPLIT"
printf '%s\n' "$OUTPUT" > "$MS_ROOT/latest_fair_relay_${SPLIT}.txt"
echo "[DONE] $OUTPUT"
