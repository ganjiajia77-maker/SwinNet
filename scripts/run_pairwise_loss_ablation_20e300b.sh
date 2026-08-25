#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-/home/gjj/Swin-Unet-main/data1}
PRETRAIN_CKPT=${PRETRAIN_CKPT:-/home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth}
BASE=${BASE:-data1_pairwise_lossabl_300b20e_20260825}

COMMON_ARGS=(
  --cfg configs/swin_tiny_patch4_window7_224_lite.yaml
  --root_path "$ROOT_PATH"
  --output_dir ./model_out
  --pretrain_ckpt "$PRETRAIN_CKPT"
  --structure_profile stage23_boundary_0626
  --stage_topology_stages none
  --img_size 256
  --source_patch_size 1024
  --direct_resize_train
  --val_interval 5
  --batch_size 4
  --num_workers 4
  --max_epochs 20
  --max_train_batches 300
  --warmup_epochs 3
  --threshold 0.2
  --seed 1234
  --new_lr 1e-4
  --pretrained_lr 3e-5
  --enable_highres_structure_stream
  --highres_structure_fuse_stages stage23
  --highres_structure_fusion_mode stage23
  --highres_structure_skeleton_weight 0.008
  --stage2_skeleton_weight 0.008
  --stage3_skeleton_weight 0.012
  --stage_connectivity_factor 0.5
  --stage_direction_factor 0.2
  --surface_loss bce_dice
  --masked_connectivity_center_experiment
  --no-use_ema
)

echo "==== Runtime pairwise connectivity check ===="
OPENCV_LOG_LEVEL=ERROR python check_pairwise_connectivity_runtime.py \
  --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
  --img_size 256 \
  --structure_profile stage23_boundary_0626 \
  --enable_highres_structure_stream \
  --highres_structure_fuse_stages stage23 \
  --highres_structure_fusion_mode stage23

verify_checkpoint_pairwise() {
  local run="$1"
  python - "$run" <<'PY'
import sys
import torch

run = sys.argv[1]
ckpt = f"./model_out/{run}/best.pth"
state = torch.load(ckpt, map_location="cpu")["model_state_dict"]
has_pairwise = any(".connectivity_head.edge_mlp." in key for key in state)
has_axis = any(".connectivity_head.axis_basis" in key for key in state)
has_old = any(key.endswith(".connectivity_head.weight") for key in state)
print(f"checkpoint pairwise edge_mlp: {has_pairwise}")
print(f"checkpoint axis_basis: {has_axis}")
print(f"checkpoint old conv: {has_old}")
if not has_pairwise or not has_axis or has_old:
    raise SystemExit(f"{ckpt} is not a pairwise-head checkpoint")
PY
}

run_one() {
  local name="$1"
  shift
  local run="${BASE}_${name}"

  echo "==== TRAIN ${run} ===="
  OPENCV_LOG_LEVEL=ERROR PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python train_image.py \
      "${COMMON_ARGS[@]}" \
      --run_name "$run" \
      "$@"

  echo "==== VERIFY ${run} ===="
  verify_checkpoint_pairwise "$run"

  echo "==== DIAG ${run} ===="
  OPENCV_LOG_LEVEL=ERROR python diagnose_connectivity_direction_quality.py \
    --root_path "$ROOT_PATH" \
    --model_path "./model_out/${run}/best.pth" \
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
    --threshold_sweep_end 0.15 \
    --threshold_sweep_step 0.005 \
    --connectivity_ablation full \
    | tee "./model_out/${run}/connectivity_topology_diag.txt"
}

run_one full
run_one no_highres_skel --highres_structure_skeleton_weight 0
run_one no_direction --stage_direction_factor 0
run_one no_boundary --boundary_weight 0

echo "==== Summary ===="
for name in full no_highres_skel no_direction no_boundary; do
  run="${BASE}_${name}"
  echo "================ ${run} ================"
  grep -E "8-dir macro F1|GT connection=1 mean|GT connection=0 mean|AUROC|AUPRC|Reciprocal symmetry error|Best threshold sweep|Surface clDice|Surface fragmentation|Surface short components|Connectivity graph fragmentation" \
    "./model_out/${run}/connectivity_topology_diag.txt" || true
done
