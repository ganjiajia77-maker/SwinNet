#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA:-/home/gjj/Swin-Unet-main/data1}"
RESULTS="${RESULTS:-/home/gjj/SegRoadv2-results}"
RUN="${RUN:-segroadv2_b2_data1_random512_ema0999_fp32_100e_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULTS"
cd "$REPO"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OPENCV_LOG_LEVEL=ERROR \
python -u train_data1.py \
  --root_path "$DATA" --output_dir "$RESULTS" --run_name "$RUN" \
  --pretrain_ckpt "$REPO/weights/segformer_b2_backbone_weights.pth" --phi b2 \
  --source_patch_size 1024 --img_size 512 --overlap_stride 256 \
  --batch_size 2 --num_workers 4 --max_epochs 100 --val_interval 5 \
  --lr 1e-4 --min_lr 1e-6 --weight_decay 0.01 --seed 1234 \
  --use_ema --ema_decay 0.999 --val_threshold 0.5 \
  --connectivity_targets source_fixed --prediction_mode source_fusion \
  2>&1 | tee "$RESULTS/${RUN}.log"
