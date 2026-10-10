#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?Usage: run_server.sh train|resume|sweep|test}
: "${RUN_DIR:?Export RUN_DIR to a unique experiment directory first}"
DATA_ROOT=${DATA_ROOT:-/home/gjj/Swin-Unet-main/data1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export OPENCV_LOG_LEVEL=ERROR
export PYTHONUNBUFFERED=1

if [[ "$MODE" == train || "$MODE" == resume ]]; then
  mkdir -p "$RUN_DIR"
  extra=()
  if [[ -n "${ENCODER_CKPT:-}" && "$MODE" == train ]]; then
    extra+=(--encoder_ckpt "$ENCODER_CKPT")
  fi
  if [[ "$MODE" == resume ]]; then
    extra+=(--resume "$RUN_DIR/last.pth")
  fi
  python -u train_data1.py --data_root "$DATA_ROOT" --output_dir "$RUN_DIR" \
    --epochs 80 --batch_size 1 --workers 4 --lr 2e-4 \
    --tile_size 512 --stride 256 --val_interval 5 --val_threshold 0.2 --seed 1234 \
    "${extra[@]}" 2>&1 | tee -a "$RUN_DIR/$MODE.log"
elif [[ "$MODE" == sweep ]]; then
  python -u eval_data1.py --data_root "$DATA_ROOT" --checkpoint "$RUN_DIR/best.pth" \
    --split val --tile_size 512 --stride 256 --output_dir "$RUN_DIR/unified_val_predictions" \
    2>&1 | tee "$RUN_DIR/val_inference.log"
  python -u prepare_manifest.py --run_dir "$RUN_DIR" --data_root "$DATA_ROOT"
  python -u evaluation/road_comparison/compare_predictions.py \
    --manifest "$RUN_DIR/unified_manifest.json" --split val \
    --output_dir "$RUN_DIR/unified_val_metrics" --progress_every 25 \
    2>&1 | tee "$RUN_DIR/val_unified_metrics.log"
elif [[ "$MODE" == test ]]; then
  SELECTION="$RUN_DIR/unified_val_metrics/val_selection.json"
  python -u prepare_manifest.py --run_dir "$RUN_DIR" --data_root "$DATA_ROOT" --selection "$SELECTION"
  THRESHOLD=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["threshold"])' "$RUN_DIR/selected_threshold.json")
  python -u eval_data1.py --data_root "$DATA_ROOT" --checkpoint "$RUN_DIR/best.pth" \
    --split test --threshold "$THRESHOLD" --tile_size 512 --stride 256 \
    --output_dir "$RUN_DIR/unified_test_predictions" 2>&1 | tee "$RUN_DIR/test_inference.log"
  python -u evaluation/road_comparison/compare_predictions.py \
    --manifest "$RUN_DIR/unified_manifest.json" --split test --selection "$SELECTION" \
    --output_dir "$RUN_DIR/unified_test_metrics" --progress_every 25 \
    2>&1 | tee "$RUN_DIR/test_unified_metrics.log"
  printf '\nUnified segmentation and topology result:\n'
  cat "$RUN_DIR/unified_test_metrics/test_comparison.csv"
else
  echo "Unknown mode: $MODE" >&2
  exit 2
fi
