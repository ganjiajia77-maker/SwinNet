#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${LOCAL_CS_INIT:?Specify the same original B checkpoint as training}"
: "${LOCAL_CS_PAIR:?Specify the short-run pair prefix}"
RESULTS=${RESULTS:-/home/gjj/SwinNet-roadbias/model_out}
DATA=${DATA:-/home/gjj/Swin-Unet-main/Massachusetts/tiff}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
LOCAL_CS_EVAL=${LOCAL_CS_EVAL:-$RESULTS/${LOCAL_CS_PAIR}_val_$(date +%Y%m%d_%H%M%S)}
printf '%s\n' "$LOCAL_CS_EVAL" > "$RESULTS/${LOCAL_CS_PAIR}_eval_latest.txt"
python -u scripts/evaluate_local_cs.py --initial "$LOCAL_CS_INIT" \
  --run "off=$RESULTS/${LOCAL_CS_PAIR}_off/last.pth" \
  --run "surface_only=$RESULTS/${LOCAL_CS_PAIR}_surface_only/last.pth" \
  --run "c_guided=$RESULTS/${LOCAL_CS_PAIR}_c_guided/last.pth" \
  --root_path "$DATA" \
  --fixed_gaps "${FIXED_GAPS:-$RESULTS/full_h0_joint_pair_v1/reference/fixed_gaps_and_anchors.json}" \
  --output_dir "$LOCAL_CS_EVAL"
if [[ ${RUN_TOPO:-0} == 1 ]]; then
  TOOL=${TOPO_TOOL:-/home/gjj/unified-road-six-4b7d37a}
  python -u "$TOOL/evaluation/road_comparison/compare_predictions.py" \
    --manifest "$LOCAL_CS_EVAL/manifest.json" --split val \
    --output_dir "$LOCAL_CS_EVAL/topology" --workers 4 --progress_every 1
fi
echo "[ALL DONE] $LOCAL_CS_EVAL"
