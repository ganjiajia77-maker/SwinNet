#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export NO_ENCODER_A1A2=1
export H_REASSEMBLY_PRIOR=predicted
case "${1:-}" in
  smoke) mode=pixel_smoke ;;
  train) mode=pixel_train ;;
  resume) mode=pixel_resume ;;
  *) echo 'Usage: run_hpixel_no_a1a2.sh smoke|train|resume' >&2; exit 2 ;;
esac
exec bash scripts/run_native_h0_pair.sh "$mode"
