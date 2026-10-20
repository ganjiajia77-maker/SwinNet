#!/usr/bin/env bash
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=${DATA:-/home/gjj/Swin-Unet-main/Massachusetts/tiff}
OUT=${OUT:-/home/gjj/SwinNet-roadbias/model_out}
PRETRAIN=${PRETRAIN:-/home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth}
SEED=${SEED:-1234}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export OPENCV_LOG_LEVEL=ERROR
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
cd "$REPO"
MODE=${1:-}
RUN=${RUN:-}
if [[ -z "$MODE" ]]; then
  echo "Usage: bash scripts/run_e202_surface_gap_repair_mass1500.sh train|resume|sweep|test|topology|baseline-only" >&2
  exit 2
fi
mkdir -p "$OUT"
if [[ "$MODE" == train || "$MODE" == baseline-only ]]; then
  RUN=${RUN:-e202_gap_online512_v2_mass1500_fp32_joint80_s${SEED}_$(date +%Y%m%d_%H%M%S)}
  if [[ "$MODE" == baseline-only ]]; then RUN=${RUN}_baseline_only; fi
  test ! -e "$OUT/$RUN" || { echo "Run exists: $OUT/$RUN" >&2; exit 2; }
  printf '%s\n' "$RUN" > "$OUT/latest_e202_surface_gap_repair_run.txt"
  ARGS=(train)
  [[ "$MODE" == baseline-only ]] && ARGS=(baseline-only)
  python -u surface_gap_repair_experiment.py "${ARGS[@]}" --gpu 0 --seed "$SEED" \
    --root_path "$DATA" --output_dir "$OUT" --run_name "$RUN" --pretrain_ckpt "$PRETRAIN" \
    --max_epochs 80 --lambda_repair 1.0 --lambda_cldice 0.1
  exit 0
fi
if [[ -z "$RUN" ]]; then
  test -s "$OUT/latest_e202_surface_gap_repair_run.txt" || { echo "Set RUN or train first" >&2; exit 2; }
  RUN=$(<"$OUT/latest_e202_surface_gap_repair_run.txt")
fi
RUN_DIR="$OUT/$RUN"
if [[ "$MODE" == resume ]]; then
  python -u surface_gap_repair_experiment.py resume --gpu 0 --seed "$SEED" \
    --root_path "$DATA" --output_dir "$OUT" --run_name "$RUN" --pretrain_ckpt "$PRETRAIN" \
    --checkpoint "$RUN_DIR/last.pth" --max_epochs 80
  exit 0
fi
CKPT=${CHECKPOINT:-$RUN_DIR/best.pth}
test -s "$CKPT" || { echo "Missing checkpoint: $CKPT" >&2; exit 2; }
COMMON=(--gpu 0 --root_path "$DATA" --output_dir "$OUT" --run_name "$RUN" --checkpoint "$CKPT" --pretrain_ckpt "$PRETRAIN")
case "$MODE" in
  sweep) python -u surface_gap_repair_experiment.py sweep "${COMMON[@]}" ;;
  test) python -u surface_gap_repair_experiment.py test "${COMMON[@]}" ;;
  topology) python -u surface_gap_repair_experiment.py topology "${COMMON[@]}" ;;
  *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
