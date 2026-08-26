#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-/home/gjj/Swin-Unet-main/data1}
PRETRAIN_CKPT=${PRETRAIN_CKPT:-/home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth}
RUN=${RUN:-data1_d0168c7_lossvars_stage23_1024to256_60e_20260826}
STAGE_DIRECTION_FACTOR=${STAGE_DIRECTION_FACTOR:-0.1}

echo "==== Code version ===="
git log -1 --oneline

echo "==== Runtime connectivity head check: d0168c7 old conv expected ===="
python - <<'PY'
from config import get_config
from networks.vision_transformer import SwinUnet
import argparse

args = argparse.Namespace(
    cfg="configs/swin_tiny_patch4_window7_224_lite.yaml",
    opts=None,
    zip=False,
    cache_mode="",
    resume="",
    accumulation_steps=0,
    use_checkpoint=False,
    amp_opt_level="",
    tag="",
    eval=False,
    throughput=False,
    dataset="ImageData",
    n_class=2,
    batch_size=4,
)
config = get_config(args)
model = SwinUnet(
    config=config,
    img_size=256,
    num_classes=1,
    return_skeleton=True,
    structure_profile="stage23_boundary_0626",
    enable_highres_structure_stream=True,
    highres_structure_fuse_stages="stage23",
    highres_structure_fusion_mode="stage23",
)
state = model.state_dict()
print("pairwise edge_mlp:", any(".connectivity_head.edge_mlp." in key for key in state))
print("prior_embed:", any(".connectivity_head.prior_embed." in key for key in state))
print("old conv:", any(key.endswith(".connectivity_head.weight") for key in state))
PY

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
  --no-use_ema

echo "==== VERIFY ${RUN} checkpoint head ===="
python - "$RUN" <<'PY'
import sys
import torch

run = sys.argv[1]
ckpt = f"./model_out/{run}/best.pth"
state = torch.load(ckpt, map_location="cpu")["model_state_dict"]
print("pairwise edge_mlp:", any(".connectivity_head.edge_mlp." in key for key in state))
print("prior_embed:", any(".connectivity_head.prior_embed." in key for key in state))
print("old conv:", any(key.endswith(".connectivity_head.weight") for key in state))
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
  --img_size 256 \
  --source_patch_size 1024 \
  --threshold 0.04 \
  | tee "./model_out/${RUN}/connectivity_topology_diag_best.txt"
