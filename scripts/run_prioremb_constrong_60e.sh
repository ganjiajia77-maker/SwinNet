#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-/home/gjj/Swin-Unet-main/data1}
PRETRAIN_CKPT=${PRETRAIN_CKPT:-/home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth}
RUN=${RUN:-data1_prioremb_constrong_simploss_stage23_1024to256_60e_20260826}
STAGE_DIRECTION_FACTOR=${STAGE_DIRECTION_FACTOR:-0.1}

echo "==== Runtime pairwise prior-embedding check ===="
OPENCV_LOG_LEVEL=ERROR python check_pairwise_connectivity_runtime.py \
  --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
  --img_size 256 \
  --structure_profile stage23_boundary_0626 \
  --enable_highres_structure_stream \
  --highres_structure_fuse_stages stage23 \
  --highres_structure_fusion_mode stage23

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
  --val_interval 1 \
  --batch_size 4 \
  --num_workers 4 \
  --max_epochs 60 \
  --warmup_epochs 10 \
  --threshold 0.2 \
  --seed 1234 \
  --new_lr 2e-4 \
  --pretrained_lr 5e-5 \
  --enable_highres_structure_stream \
  --highres_structure_fuse_stages stage23 \
  --highres_structure_fusion_mode stage23 \
  --highres_structure_skeleton_weight 0 \
  --stage2_skeleton_weight 0.008 \
  --stage3_skeleton_weight 0.012 \
  --stage_connectivity_factor 2.0 \
  --stage_direction_factor "$STAGE_DIRECTION_FACTOR" \
  --boundary_weight 0.005 \
  --connectivity_pos_weight 5 \
  --connectivity_focal_gamma 1.5 \
  --surface_loss bce_dice \
  --masked_connectivity_center_experiment \
  --no-use_ema

echo "==== VERIFY ${RUN} checkpoint head ===="
python - "$RUN" <<'PY'
import sys
import torch

run = sys.argv[1]
ckpt = f"./model_out/{run}/best.pth"
state = torch.load(ckpt, map_location="cpu")["model_state_dict"]
has_pairwise = any(".connectivity_head.edge_mlp." in key for key in state)
has_axis = any(".connectivity_head.axis_basis" in key for key in state)
has_prior = any(".connectivity_head.prior_embed." in key for key in state)
has_old = any(key.endswith(".connectivity_head.weight") for key in state)
print(f"pairwise edge_mlp: {has_pairwise}")
print(f"axis_basis: {has_axis}")
print(f"prior_embed: {has_prior}")
print(f"old conv: {has_old}")
key = "swin_unet.decoder_structure_blocks.2.connectivity_head.edge_mlp.0.weight"
if key in state:
    print(key, tuple(state[key].shape))
if not has_pairwise or not has_axis or not has_prior or has_old:
    raise SystemExit(f"{ckpt} is not the expected prior-embedding pairwise checkpoint")
PY

echo "==== DIAG ${RUN} best.pth ===="
OPENCV_LOG_LEVEL=ERROR python diagnose_connectivity_direction_quality.py \
  --root_path "$ROOT_PATH" \
  --model_path "./model_out/${RUN}/best.pth" \
  --split val \
  --max_batches 50 \
  --batch_size 4 \
  --num_workers 4 \
  --stage stage3_refine \
  --structure_profile stage23_boundary_0626 \
  --model_impl standard \
  --img_size 256 \
  --source_patch_size 1024 \
  --threshold 0.04 \
  --threshold_sweep_start 0.01 \
  --threshold_sweep_end 0.20 \
  --threshold_sweep_step 0.005 \
  --connectivity_ablation full \
  | tee "./model_out/${RUN}/connectivity_topology_diag_best.txt"
