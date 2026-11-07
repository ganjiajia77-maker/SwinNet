#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${LOCAL_CS_INIT:?Specify the full original B checkpoint path}"
export RESULTS=${RESULTS:-/home/gjj/SwinNet-roadbias/model_out}
export LOCAL_CS_PAIR=${LOCAL_CS_PAIR:-h0_local_cs_short_$(date +%Y%m%d_%H%M%S)}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
mode=${1:-all}
case "$mode" in
  all) modes=(off surface_only c_guided) ;;
  off|surface_only|c_guided) modes=("$mode") ;;
  *) echo 'Usage: run_local_cs_short.sh all|off|surface_only|c_guided' >&2; exit 2 ;;
esac
for item in "${modes[@]}"; do
  export LOCAL_CS_MODE=$item LOCAL_CS_RUN="${LOCAL_CS_PAIR}_${item}"
  bash scripts/run_native_h0_pair.sh local_cs
done
python scripts/summarize_local_cs.py --results "$RESULTS" --pair "$LOCAL_CS_PAIR"
