#!/usr/bin/env bash
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${CKPT:?Set CKPT to the checkpoint used for the completed gate comparison}"
: "${DATA:?Set DATA to the corresponding dataset root}"
: "${PREVIOUS_OUT:?Set PREVIOUS_OUT to the completed gate comparison directory}"
GPU=${GPU:-0}
OUT=${OUT:-$PREVIOUS_OUT/writeback_effect_$(date +%Y%m%d_%H%M%S)}
SUMMARY="$PREVIOUS_OUT/comparison/summary.json"
export CUDA_VISIBLE_DEVICES=$GPU
export OPENCV_LOG_LEVEL=ERROR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"

test -s "$CKPT" || { echo "Missing checkpoint: $CKPT" >&2; exit 1; }
test -s "$SUMMARY" || { echo "Missing completed comparison: $SUMMARY" >&2; exit 1; }
test ! -e "$OUT" || { echo "Output already exists: $OUT" >&2; exit 1; }
grep -Fq "$CKPT" "$PREVIOUS_OUT/original_inference.log" || { echo "Checkpoint path differs from the original inference log" >&2; exit 1; }
INFO=$(python -c 'import sys,json,torch; c=torch.load(sys.argv[1],map_location="cpu",weights_only=False); a=c.get("args",{}); s=json.load(open(sys.argv[2])); a.get("enable_global_topology",False) or sys.exit("Checkpoint has no active global topology"); native=a.get("source_patch_size",1024); native==s["protocol"]["image_size"] or sys.exit("Native image size differs from completed comparison"); print(a.get("img_size",512),native,int(bool(a.get("use_ema",False))),s["threshold"],s["split"])' "$CKPT" "$SUMMARY")
read -r IMAGE_SIZE SOURCE_SIZE USE_EMA THRESHOLD SPLIT <<< "$INFO"
EMA_ARGS=()
if [[ "$USE_EMA" = 1 ]]; then EMA_ARGS+=(--require_ema); fi
mkdir -p "$OUT"
echo "checkpoint=$CKPT split=$SPLIT threshold=$THRESHOLD tile=$IMAGE_SIZE native=$SOURCE_SIZE EMA=$USE_EMA GPU=$GPU output=$OUT"

python -u test_image.py \
  --root_path "$DATA" --model_path "$CKPT" \
  --cfg "$REPO/configs/swin_tiny_patch4_window7_224_lite.yaml" \
  --split "$SPLIT" --img_size "$IMAGE_SIZE" --source_patch_size "$SOURCE_SIZE" \
  --overlap_infer --save_overlap_logits --threshold "$THRESHOLD" \
  --diagnostic_zero_global_writeback "${EMA_ARGS[@]}" \
  --output_dir "$OUT/zero" \
  2>&1 | tee "$OUT/zero_inference.log"

python -u scripts/diagnose_h0_writeback_effect.py \
  --baseline_dir "$PREVIOUS_OUT/original" \
  --floor_dir "$PREVIOUS_OUT/floor_0p2" \
  --zero_dir "$OUT/zero" \
  --root_path "$DATA" --split "$SPLIT" \
  --source_patch_size "$SOURCE_SIZE" --threshold "$THRESHOLD" \
  --output_dir "$OUT/analysis" \
  2>&1 | tee "$OUT/effect_analysis.log"

echo "Results: $OUT/analysis/summary.json"
