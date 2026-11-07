#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${B_CHECKPOINT:?Specify the original native B checkpoint explicitly}"
: "${STAGE_OUT:?Specify a new independent diagnostic output directory}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export OPENCV_LOG_LEVEL=ERROR
test -s "$B_CHECKPOINT"
python -u scripts/diagnose_b_stage_features.py \
  --checkpoint "$B_CHECKPOINT" \
  --root_path "${DATA:-/home/gjj/Swin-Unet-main/Massachusetts/tiff}" \
  --fixed_gaps "${FIXED_GAPS:-/home/gjj/SwinNet-roadbias/model_out/full_h0_joint_pair_v1/reference/fixed_gaps_and_anchors.json}" \
  --output_dir "$STAGE_OUT" \
  --train_images "${PROBE_TRAIN_IMAGES:-64}" --holdout_images "${PROBE_HOLDOUT_IMAGES:-16}" \
  --crops_per_image "${PROBE_CROPS_PER_IMAGE:-2}" --probe_epochs "${PROBE_EPOCHS:-3}" \
  --normalization_crops "${PROBE_NORMALIZATION_CROPS:-32}" \
  --case_panels "${STAGE_CASE_PANELS:-24}" --threshold "${STAGE_THRESHOLD:-0.2}"
python scripts/show_b_stage_features.py --output_dir "$STAGE_OUT" > "$STAGE_OUT/TABLES.txt"
echo "[ALL DONE] $STAGE_OUT/index.html"
