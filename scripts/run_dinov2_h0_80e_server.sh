#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OPENCV_LOG_LEVEL=ERROR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODE="${1:-train}"
DATA_ROOT="${DATA_ROOT:-/home/gjj/Swin-Unet-main/data1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/gjj/SwinNet-roadbias/model_out}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-/home/gjj/Swin-Unet-main/pretrained_ckpt/dinov2_converted.pth}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
case "$MICRO_BATCH_SIZE" in
  1) ACCUMULATION_STEPS=4 ;;
  2) ACCUMULATION_STEPS=2 ;;
  4) ACCUMULATION_STEPS=1 ;;
  *) echo "MICRO_BATCH_SIZE must be 1, 2, or 4 (effective batch size is 4)"; exit 2 ;;
esac
if [[ "$MODE" == train ]]; then
  RUN="${RUN:-data1_h0_dinov2_l16_random512_fp32_ema80e_$(date +%Y%m%d_%H%M%S)}"
else
  : "${RUN:?Set RUN to the existing experiment name}"
fi
RUN_DIR="$OUTPUT_ROOT/$RUN"
COMMON=(
  --root_path "$DATA_ROOT" --output_dir "$OUTPUT_ROOT" --run_name "$RUN"
  --cfg configs/dinov2_l16_h0_512.yaml --encoder_type dinov2_l16 --freeze_pretrained_encoder
  --pretrain_ckpt "$PRETRAIN_CKPT"
  --structure_profile stage23_boundary_0626 --stage_skeleton_mode direct
  --enable_highres_structure_stream --highres_structure_channels 64
  --highres_structure_fuse_stages stage23 --highres_structure_fusion_mode stage23
  --enable_h3_surface_fusion --remove_stage2_pre_topology_source
  --enable_global_topology --global_topology_max_nodes 32 --global_topology_heads 4
  --global_topology_alpha_max 0.05
  --img_size 512 --source_patch_size 1024 --random_crop_train --random_crops_per_image 1
  --overlap_stride 256 --batch_size "$MICRO_BATCH_SIZE" --accumulation_steps "$ACCUMULATION_STEPS"
  --max_epochs 80 --warmup_epochs 10
  --val_interval 5 --num_workers 4 --disable_persistent_workers
  --base_lr 5e-4 --min_lr 1e-5 --pretrained_lr 5e-5 --pretrained_min_lr 5e-6
  --new_lr 2e-4 --new_min_lr 1e-5 --threshold 0.2 --skeleton_threshold 0.5
  --stage2_skeleton_weight 0.008 --stage3_skeleton_weight 0.012 --skeleton_pos_weight 2.0
  --stage_skeleton_loss_factor 1.0 --stage_skeleton_only_loss_factor 1.0
  --stage_connectivity_factor 3.0 --stage_direction_factor 0.2
  --masked_connectivity_center_experiment --connectivity_pos_weight 2.0
  --connectivity_focal_gamma 1.5 --edge_contrastive_margin 0.1
  --directional_pos_weight_cardinal 1.0 --directional_pos_weight_diagonal 2.5
  --road_attention_weight 0.0 --highres_structure_skeleton_weight 0.0 --final_skeleton_weight 0.0
  --surface_focal_gamma 1.0 --amp_dtype none --seed 1234 --deterministic 1
  --use_ema --ema_decay 0.999
)
echo "[RUN] $RUN_DIR; physical CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
case "$MODE" in
  train)
    [[ -s "$PRETRAIN_CKPT" ]] || { echo "Missing converted weights: $PRETRAIN_CKPT"; exit 1; }
    [[ ! -e "$RUN_DIR" ]] || { echo "Experiment already exists; use resume or a new RUN"; exit 1; }
    mkdir -p "$OUTPUT_ROOT"
    python -u train_image.py "${COMMON[@]}" 2>&1 | tee "$OUTPUT_ROOT/${RUN}_console.log"
    ;;
  resume)
    [[ -s "$RUN_DIR/last.pth" ]] || { echo "Missing $RUN_DIR/last.pth"; exit 1; }
    python -u train_image.py "${COMMON[@]}" --resume "$RUN_DIR/last.pth" 2>&1 | tee -a "$RUN_DIR/resume.log"
    ;;
  sweep)
    python -u eval_dinov2_h0.py --root_path "$DATA_ROOT" --checkpoint "$RUN_DIR/best.pth" \
      --split val --output_dir "$RUN_DIR/threshold_val" 2>&1 | tee "$RUN_DIR/threshold_val.log"
    ;;
  test)
    python -u eval_dinov2_h0.py --root_path "$DATA_ROOT" --checkpoint "$RUN_DIR/best.pth" \
      --split test --threshold_file "$RUN_DIR/threshold_val/best_threshold.json" \
      --output_dir "$RUN_DIR/test_selected_threshold" 2>&1 | tee "$RUN_DIR/test.log"
    ;;
  *) echo "Usage: bash scripts/run_dinov2_h0_80e_server.sh {train|resume|sweep|test}"; exit 2 ;;
esac
