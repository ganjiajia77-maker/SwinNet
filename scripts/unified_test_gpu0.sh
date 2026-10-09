#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="${RESULTS:-/home/gjj/SegRoadv2-results}"
RUN="${RUN:-$(cat "$RESULTS/latest_segroadv2_run.txt")}"
RUN_DIR="$RESULTS/$RUN"
AUDIT="${AUDIT:-$(cat "$RUN_DIR/latest_unified_audit.txt")}"
cd "$REPO"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OPENCV_LOG_LEVEL=ERROR \
python -u unified_eval_data1.py --split test --output_dir "$AUDIT" \
  2>&1 | tee "$RUN_DIR/$(basename "$AUDIT")_test.log"
cat "$AUDIT/test/test_comparison.csv"
