#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA:-/home/gjj/Swin-Unet-main/data1}"
RESULTS="${RESULTS:-/home/gjj/SegRoadv2-results}"
RUN="${RUN:-$(cat "$RESULTS/latest_segroadv2_run.txt")}"
RUN_DIR="$RESULTS/$RUN"
CKPT="$RUN_DIR/best.pth"
AUDIT="${AUDIT:-$RUN_DIR/unified_metrics_$(date +%Y%m%d_%H%M%S)}"
test -s "$CKPT" || { echo "Checkpoint missing: $CKPT" >&2; exit 1; }
cd "$REPO"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OPENCV_LOG_LEVEL=ERROR \
python -u unified_eval_data1.py \
  --split val --root_path "$DATA" --model_path "$CKPT" --output_dir "$AUDIT" \
  --source_patch_size 1024 --tile_size 512 --overlap_stride 256 \
  --weights ema --prediction_mode source_fusion \
  --thresholds 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95 \
  2>&1 | tee "$RUN_DIR/$(basename "$AUDIT")_val.log"
printf '%s\n' "$AUDIT" > "$RUN_DIR/latest_unified_audit.txt"
cat "$AUDIT/val/val_threshold_scores.csv"
cat "$AUDIT/val/val_selection.json"
cat "$AUDIT/val/val_comparison.csv"
