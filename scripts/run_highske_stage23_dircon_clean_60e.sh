#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-/home/gjj/SwinNet-roadbias/data1}
PRETRAIN_CKPT=${PRETRAIN_CKPT:-/home/gjj/SwinNet-roadbias/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth}
RUN=${RUN:-data1_highske_stage23_dircon_clean_w008_direct256_60e_20260827}
NUM_WORKERS=${NUM_WORKERS:-4}

echo "==== GRADIENT DIAG ${RUN} ===="
OPENCV_LOG_LEVEL=ERROR PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python diagnose_gradient_conflict.py \
  --root_path "$ROOT_PATH" \
  --split train \
  --batch_size 1 \
  --num_workers "$NUM_WORKERS" \
  --batch_index 0 \
  --img_size 256 \
  --source_patch_size 1024 \
  --structure_profile stage23_boundary_0626 \
  --stage_topology_stages none \
  --disable_msfe_skip \
  --enable_highres_structure_stream \
  --highres_structure_fuse_stages stage23 \
  --highres_structure_fusion_mode stage23 \
  --highres_structure_skeleton_weight 0.008 \
  --stage2_skeleton_weight 0.008 \
  --stage3_skeleton_weight 0.012 \
  --stage_connectivity_factor 1.0 \
  --stage_direction_factor 0.2 \
  --masked_connectivity_center_experiment

echo "==== TRAIN ${RUN} ===="
OPENCV_LOG_LEVEL=ERROR PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_image.py \
  --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
  --root_path "$ROOT_PATH" \
  --output_dir ./model_out \
  --run_name "$RUN" \
  --pretrain_ckpt "$PRETRAIN_CKPT" \
  --structure_profile stage23_boundary_0626 \
  --stage_topology_stages none \
  --img_size 256 \
  --source_patch_size 1024 \
  --direct_resize_train \
  --batch_size 4 \
  --num_workers "$NUM_WORKERS" \
  --max_epochs 60 \
  --warmup_epochs 10 \
  --val_interval 6 \
  --threshold 0.2 \
  --skeleton_threshold 0.5 \
  --seed 1234 \
  --new_lr 2e-4 \
  --pretrained_lr 5e-5 \
  --disable_msfe_skip \
  --enable_highres_structure_stream \
  --highres_structure_fuse_stages stage23 \
  --highres_structure_fusion_mode stage23 \
  --highres_structure_skeleton_weight 0.008 \
  --stage2_skeleton_weight 0.008 \
  --stage3_skeleton_weight 0.012 \
  --stage_connectivity_factor 1.0 \
  --stage_direction_factor 0.2 \
  --surface_loss bce_dice \
  --masked_connectivity_center_experiment \
  --no-use_ema

echo "==== DIAG ${RUN} best.pth ===="
OPENCV_LOG_LEVEL=ERROR python diagnose_connectivity_direction_quality.py \
  --root_path "$ROOT_PATH" \
  --model_path "./model_out/${RUN}/best.pth" \
  --split val \
  --max_batches 50 \
  --batch_size 4 \
  --num_workers "$NUM_WORKERS" \
  --stage stage3_refine \
  --structure_profile stage23_boundary_0626 \
  --model_impl standard \
  --img_size 256 \
  --source_patch_size 1024 \
  --threshold 0.2 \
  --threshold_sweep_start 0.01 \
  --threshold_sweep_end 0.50 \
  --threshold_sweep_step 0.01 \
  --connectivity_ablation full \
  | tee "./model_out/${RUN}/connectivity_topology_diag_best.txt"
