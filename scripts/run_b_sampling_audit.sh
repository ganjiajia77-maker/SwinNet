#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${B_CHECKPOINT:?Specify the same original B checkpoint}"
: "${STAGE_SOURCE:?Specify the completed stage feature output directory}"
: "${AUDIT_OUT:?Specify a new output directory}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export OPENCV_LOG_LEVEL=ERROR
python -u scripts/diagnose_b_sampling_audit.py \
  --checkpoint "$B_CHECKPOINT" --source_dir "$STAGE_SOURCE" \
  --root_path "${DATA:-/home/gjj/Swin-Unet-main/Massachusetts/tiff}" \
  --fixed_gaps "${FIXED_GAPS:-/home/gjj/SwinNet-roadbias/model_out/full_h0_joint_pair_v1/reference/fixed_gaps_and_anchors.json}" \
  --output_dir "$AUDIT_OUT" --extra_probe_epochs "${AUDIT_PROBE_EPOCHS:-2}"
