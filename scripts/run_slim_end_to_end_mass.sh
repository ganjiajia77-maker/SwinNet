#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export OPENCV_LOG_LEVEL=ERROR
DATA=${DATA:-/home/gjj/Swin-Unet-main/Massachusetts/tiff}
RESULTS=${RESULTS:-/home/gjj/SwinNet-roadbias/model_out}
PRETRAIN=${PRETRAIN:-/home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth}
MODE=${1:-all}
[[ "$MODE" == all || "$MODE" == train || "$MODE" == resume || "$MODE" == smoke || "$MODE" == sweep || "$MODE" == test || "$MODE" == topology || "$MODE" == results ]] || { echo 'Usage: all|train|resume|smoke|sweep|test|topology|results' >&2; exit 2; }
mkdir -p "$RESULTS"
POINTER="$RESULTS/latest_slim_end_to_end_mass_run.txt"
if [[ -z ${RUN_DIR:-} ]]; then
  if [[ "$MODE" == train ]]; then
    RUN_DIR="$RESULTS/slim_e2e_mass1500_random512_fp32_noema80_b2_acc1_s1234_$(date +%Y%m%d_%H%M%S)"
  elif [[ -s "$POINTER" ]]; then
    RUN_DIR=$(cat "$POINTER")
  elif [[ "$MODE" == all || "$MODE" == smoke ]]; then
    RUN_DIR="$RESULTS/slim_e2e_mass1500_random512_fp32_noema80_b2_acc1_s1234_$(date +%Y%m%d_%H%M%S)"
  else
    echo 'Missing run pointer; set RUN_DIR.' >&2; exit 1
  fi
fi
mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > "$POINTER"
echo "RUN_DIR=$RUN_DIR; GPU=$CUDA_VISIBLE_DEVICES; FP32; batch2/acc1; no EMA"
if [[ "$MODE" == results ]]; then
  for split in val test; do
    if [[ -s "$RUN_DIR/eval/$split/comparison.csv" ]]; then
      echo "$split: $RUN_DIR/eval/$split/comparison.csv"
      cat "$RUN_DIR/eval/$split/comparison.csv"
    fi
  done
  exit 0
fi
if [[ "$MODE" == topology && -s "$RUN_DIR/eval/test/comparison.csv" ]]; then
  cat "$RUN_DIR/eval/test/comparison.csv"
  exit 0
fi
exec 9>"$RUN_DIR/.run.lock"
flock -n 9 || { echo "Already running: $RUN_DIR" >&2; exit 1; }
args=("$MODE" --root_path "$DATA" --output_dir "$RUN_DIR" --pretrain_ckpt "$PRETRAIN" --device cuda --epochs "${EPOCHS:-80}" --num_workers "${NUM_WORKERS:-4}" --seed 1234 --lambda_choice "${LAMBDA_CHOICE:-0.05}")
if [[ ${ACTIVATION_CHECKPOINT:-0} == 1 ]]; then args+=(--activation_checkpoint); fi
python -u run_slim_end_to_end_mass.py "${args[@]}" 2>&1 | tee -a "$RUN_DIR/console.log"
# all deliberately finishes with VAL selection; test remains an explicit final action.
