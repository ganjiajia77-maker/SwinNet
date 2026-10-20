#!/usr/bin/env bash
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${CKPT:?Set CKPT to the trained H0 best.pth used by the original evaluation}"
: "${DATA:?Set DATA to the checkpoint validation dataset root}"
: "${THRESHOLD:?Set THRESHOLD to the original validation-selected threshold}"
GPU=${GPU:-0}
SPLIT=${SPLIT:-val}
OUT=${OUT:-$(dirname "$CKPT")/gate_floor_diag_$(date +%Y%m%d_%H%M%S)}
export CUDA_VISIBLE_DEVICES=$GPU
export OPENCV_LOG_LEVEL=ERROR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
test -s "$CKPT" || { echo "Missing checkpoint: $CKPT" >&2; exit 1; }
test -d "$DATA/$SPLIT" || { echo "Missing split data: $DATA/$SPLIT" >&2; exit 1; }
test ! -e "$OUT" || { echo "Output already exists: $OUT" >&2; exit 1; }

INFO=$(python -c 'import sys,torch; c=torch.load(sys.argv[1],map_location="cpu",weights_only=False); a=c.get("args",{}); enabled=bool(a.get("enable_global_topology",False)); enabled or sys.exit("Checkpoint has no active global topology writeback"); variant="a" if a.get("ablate_h0_stream",False) else ("b" if a.get("static_h0_stage_fusion",False) else "full"); print(a.get("img_size",512),a.get("source_patch_size",1024),int(bool(a.get("use_ema",False))),variant)' "$CKPT")
read -r IMAGE_SIZE SOURCE_SIZE USE_EMA H0_VARIANT <<< "$INFO"
EMA_OPTION=
if [ "$USE_EMA" = 1 ]; then EMA_OPTION=--require_ema; fi
echo "checkpoint=$CKPT data=$DATA split=$SPLIT tile=$IMAGE_SIZE native=$SOURCE_SIZE threshold=$THRESHOLD EMA=$USE_EMA H0_variant=$H0_VARIANT GPU=$GPU output=$OUT"
mkdir -p "$OUT"

for FLOOR in 0 0.2; do
  NAME=original
  if [[ "$FLOOR" != 0 ]]; then NAME=floor_0p2; fi
  python -u test_image.py \
    --root_path "$DATA" --model_path "$CKPT" \
    --cfg "$REPO/configs/swin_tiny_patch4_window7_224_lite.yaml" \
    --split "$SPLIT" --img_size "$IMAGE_SIZE" --source_patch_size "$SOURCE_SIZE" \
    --overlap_infer --threshold "$THRESHOLD" \
    --writeback_gate_floor "$FLOOR" --save_overlap_logits \
    --output_dir "$OUT/$NAME" $EMA_OPTION \
    2>&1 | tee "$OUT/${NAME}_inference.log"
done

python -u scripts/compare_h0_gate_floor.py \
  --baseline_dir "$OUT/original" --floor_dir "$OUT/floor_0p2" \
  --root_path "$DATA" --split "$SPLIT" --source_patch_size "$SOURCE_SIZE" \
  --threshold "$THRESHOLD" --output_dir "$OUT/comparison" \
  2>&1 | tee "$OUT/comparison.log"

echo "Results: $OUT/comparison/summary.json"
