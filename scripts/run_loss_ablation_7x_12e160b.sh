#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-/home/gjj/Swin-Unet-main/data1}
PRETRAIN_CKPT=${PRETRAIN_CKPT:-/home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth}
RUN_PREFIX=${RUN_PREFIX:-abl12e160b_highske_stage23_dircon_20260827}
NUM_WORKERS=${NUM_WORKERS:-4}
MAX_EPOCHS=${MAX_EPOCHS:-12}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-2}
MAX_TRAIN_BATCHES=${MAX_TRAIN_BATCHES:-160}
VAL_INTERVAL=${VAL_INTERVAL:-3}
THRESHOLD=${THRESHOLD:-0.2}

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
  --batch_size 4
  --num_workers "$NUM_WORKERS"
  --max_epochs "$MAX_EPOCHS"
  --warmup_epochs "$WARMUP_EPOCHS"
  --max_train_batches "$MAX_TRAIN_BATCHES"
  --val_interval "$VAL_INTERVAL"
  --threshold "$THRESHOLD"
  --skeleton_threshold 0.5
  --seed 1234
  --new_lr 2e-4
  --pretrained_lr 5e-5
  --disable_msfe_skip
  --enable_highres_structure_stream
  --highres_structure_channels 64
  --highres_structure_fuse_stages stage23
  --highres_structure_fusion_mode stage23
  --surface_loss bce_dice
  --masked_connectivity_center_experiment
  --no-use_ema
)

run_one() {
  local id="$1"
  local final_ske="$2"
  local stage2="$3"
  local stage3="$4"
  local con_factor="$5"
  local dir_factor="$6"
  local high_ske="$7"
  local boundary="$8"
  local run="${RUN_PREFIX}_${id}"

  echo "==== TRAIN ${id}: ${run} ===="
  echo "final_ske=${final_ske} stage2=${stage2} stage3=${stage3} con=${con_factor} dir=${dir_factor} high=${high_ske} boundary=${boundary}"
  OPENCV_LOG_LEVEL=ERROR PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_image.py \
    "${COMMON_ARGS[@]}" \
    --run_name "$run" \
    --final_skeleton_weight "$final_ske" \
    --final_connectivity_weight 0 \
    --stage2_skeleton_weight "$stage2" \
    --stage3_skeleton_weight "$stage3" \
    --stage_connectivity_factor "$con_factor" \
    --stage_direction_factor "$dir_factor" \
    --highres_structure_skeleton_weight "$high_ske" \
    --boundary_weight "$boundary"

  echo "==== DIAG ${id}: ${run} best.pth ===="
  OPENCV_LOG_LEVEL=ERROR python diagnose_connectivity_direction_quality.py \
    --root_path "$ROOT_PATH" \
    --model_path "./model_out/${run}/best.pth" \
    --split val \
    --max_batches 50 \
    --batch_size 4 \
    --num_workers "$NUM_WORKERS" \
    --stage stage3_refine \
    --structure_profile stage23_boundary_0626 \
    --model_impl standard \
    --img_size 256 \
    --source_patch_size 1024 \
    --threshold "$THRESHOLD" \
    --threshold_sweep_start 0.01 \
    --threshold_sweep_end 0.50 \
    --threshold_sweep_step 0.01 \
    --connectivity_ablation full \
    | tee "./model_out/${run}/connectivity_topology_diag_best.txt"
}

# ID | Seg | Ske | Con | Dir | Highres | Boundary
run_one A 0.00 0.000 0.000 0.0 0.0 0.000 0.000
run_one B 0.10 0.000 0.000 0.0 0.0 0.000 0.000
run_one C 0.10 0.008 0.012 1.0 0.0 0.000 0.000
run_one D 0.10 0.008 0.012 1.0 0.2 0.000 0.000
run_one E 0.10 0.008 0.012 1.0 0.2 0.008 0.000
run_one F 0.10 0.008 0.012 1.0 0.2 0.000 0.010
run_one G 0.10 0.008 0.012 1.0 0.2 0.008 0.010
