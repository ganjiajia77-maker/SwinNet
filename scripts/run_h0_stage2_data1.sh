#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO=$PWD
DATA=${DATA:-/home/gjj/Swin-Unet-main/data1}
RESULTS=${RESULTS:-/home/gjj/SwinNet-roadbias/model_out}
PRETRAIN=${PRETRAIN:-/home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth}
: "${H0_REFERENCE:?Set H0_REFERENCE to the original Data1 H0 best.pth}"
: "${PHASE_RUN_NAME:?Set PHASE_RUN_NAME}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OPENCV_LOG_LEVEL=ERROR
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
test -s "$H0_REFERENCE"
test -s "$PRETRAIN"
test -d "$DATA/train"
test -d "$DATA/val"
run_dir="$RESULTS/$PHASE_RUN_NAME"
extra=()
case "${1:-}" in
  train)
    test ! -e "$run_dir" || { echo "Output already exists: $run_dir" >&2; exit 1; }
    ;;
  resume)
    test -s "$run_dir/last.pth"
    extra=(--resume "$run_dir/last.pth")
    ;;
  *)
    echo 'Usage: run_h0_stage2_data1.sh train|resume' >&2
    exit 2
    ;;
esac
exec python -u "$REPO/train_image.py" \
  --root_path "$DATA" --output_dir "$RESULTS" --run_name "$PHASE_RUN_NAME" \
  --cfg "$REPO/configs/swin_tiny_patch4_window7_224_lite.yaml" --pretrain_ckpt "$PRETRAIN" \
  --encoder_h_postmerge_phase --global_topology_stage2 --h0_data1 \
  --h0_config_checkpoint "$H0_REFERENCE" "${extra[@]}"
