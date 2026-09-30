#!/usr/bin/env bash
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate swinunet

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA=/home/gjj/Swin-Unet-main/data1
PRETRAIN=/home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth
MODEL_ROOT=/home/gjj/SwinNet-roadbias/model_out
CFG="$REPO/configs/swin_tiny_patch4_window7_224_lite.yaml"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OPENCV_LOG_LEVEL=ERROR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"

case "${1:-}" in
  train)
    RUN="${RUN:-data1_h2h3_p64psi_sparse_fp32_70e_$(date +%Y%m%d_%H%M%S)}"
    OUTDIR="$MODEL_ROOT/$RUN"
    LOG="$MODEL_ROOT/${RUN}.train.log"
    if [[ -e "$OUTDIR" ]]; then
      echo "Run directory already exists: $OUTDIR" >&2
      exit 1
    fi
    if [[ -e "$LOG" ]]; then
      echo "Run log already exists: $LOG" >&2
      exit 1
    fi
    [[ -d "$DATA" && -f "$PRETRAIN" ]] || {
      echo "Missing data or pretrained checkpoint" >&2
      exit 1
    }
    mkdir -p "$MODEL_ROOT"
    printf 'RUN=%s\nOUTDIR=%s\n' "$RUN" "$OUTDIR"
    python -u train_image.py \
      --root_path "$DATA" \
      --output_dir "$MODEL_ROOT" \
      --run_name "$RUN" \
      --cfg "$CFG" \
      --pretrain_ckpt "$PRETRAIN" \
      --structure_profile stage23_boundary_0626 \
      --enable_highres_structure_stream \
      --highres_structure_channels 64 \
      --highres_structure_fuse_stages stage23 \
      --highres_structure_fusion_mode stage23 \
      --highres_structure_skeleton_weight 0.0 \
      --enable_global_topology \
      --global_topology_max_nodes 32 \
      --global_topology_heads 4 \
      --global_topology_alpha_max 0.05 \
      --enable_coarse_road_mask \
      --enable_psi_directional_descriptor \
      --enable_sparse_window_compute \
      --coarse_routing_mode p64 \
      --coarse_corridor_window_radius 0 \
      --coarse_route_warmup_epochs 0 \
      --routing_warmup_epochs 10 \
      --routing_road_recall_min 0.99 \
      --routing_skeleton_recall_min 0.98 \
      --routing_threshold_candidates 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50 \
      --stage2_window_threshold 0.25 \
      --stage3_window_threshold 0.25 \
      --coarse_road_loss_weight 0.2 \
      --coarse_road_pos_weight 4.0 \
      --img_size 256 \
      --source_patch_size 1024 \
      --direct_resize_train \
      --batch_size 4 \
      --max_epochs 70 \
      --warmup_epochs 10 \
      --num_workers 4 \
      --val_interval 5 \
      --base_lr 5e-4 \
      --min_lr 1e-5 \
      --pretrained_lr 5e-5 \
      --pretrained_min_lr 5e-6 \
      --new_lr 2e-4 \
      --new_min_lr 1e-5 \
      --threshold 0.2 \
      --skeleton_threshold 0.5 \
      --stage2_skeleton_weight 0.008 \
      --stage3_skeleton_weight 0.012 \
      --skeleton_pos_weight 2.0 \
      --stage_connectivity_factor 3.0 \
      --stage_direction_factor 0.2 \
      --road_attention_weight 0.0 \
      --masked_connectivity_center_experiment \
      --connectivity_pos_weight 2.0 \
      --connectivity_focal_gamma 1.5 \
      --edge_contrastive_margin 0.1 \
      --directional_pos_weight_cardinal 1.0 \
      --directional_pos_weight_diagonal 2.5 \
      --surface_focal_gamma 1.0 \
      --amp_dtype none \
      --seed 1234 \
      --deterministic 1 \
      --no-use_ema \
      2>&1 | tee "$LOG"
    mv "$LOG" "$OUTDIR/train.log"
    ;;
  sweep|test)
    : "${RUN:?Set RUN to the completed training run name}"
    OUTDIR="$MODEL_ROOT/$RUN"
    CKPT="$OUTDIR/best.pth"
    [[ -f "$CKPT" ]] || { echo "Missing checkpoint: $CKPT" >&2; exit 1; }
    if [[ "$1" == sweep ]]; then
      python -u threshold_sweep_test.py \
        --root_path "$DATA" \
        --model_path "$CKPT" \
        --split val \
        --img_size 256 \
        --source_patch_size 1024 \
        --batch_size 4 \
        --num_workers 4 \
        --cfg "$CFG" \
        --amp_dtype none \
        2>&1 | tee "$OUTDIR/threshold_sweep_val.log"
    else
      : "${BEST_THRESHOLD:?Set BEST_THRESHOLD from the validation sweep}"
      python -u test_image.py \
        --root_path "$DATA" \
        --model_path "$CKPT" \
        --output_dir "/home/gjj/SwinNet-roadbias/predictions/${RUN}_thr${BEST_THRESHOLD}" \
        --img_size 256 \
        --source_patch_size 1024 \
        --batch_size 4 \
        --num_workers 4 \
        --cfg "$CFG" \
        --threshold "$BEST_THRESHOLD" \
        --skeleton_threshold 0.5 \
        --amp_dtype none \
        2>&1 | tee "$OUTDIR/test_thr${BEST_THRESHOLD}.log"
    fi
    ;;
  *)
    echo "Usage: bash scripts/run_p64_psi_70e_server.sh {train|sweep|test}" >&2
    exit 2
    ;;
esac
