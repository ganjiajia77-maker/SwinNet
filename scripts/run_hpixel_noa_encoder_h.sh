#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export H_ENCODER_EXCHANGE=1
exec bash scripts/run_hpixel_no_a1a2.sh "${1:-}"
