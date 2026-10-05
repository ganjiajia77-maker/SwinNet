#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OPENCV_LOG_LEVEL=ERROR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA_ROOT="${DATA_ROOT:-/home/gjj/Swin-Unet-main/data1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/gjj/SwinNet-roadbias/model_out}"
PRETRAIN="${PRETRAIN:-/home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
WORKERS="${WORKERS:-4}"
CACHE_DIR="${CACHE_DIR:-$DATA_ROOT/.anchor_reachability_cache}"
MODE="${1:-train}"
if [[ "$MODE" == cache ]]; then
  python -u prepare_anchor_reachability.py --root_path "$DATA_ROOT" --cache_dir "$CACHE_DIR" --workers "$WORKERS"
  exit 0
fi
: "${RUN:?Set RUN to the experiment folder name first}"
RUN_DIR="${RUN_DIR:-$OUTPUT_ROOT/$RUN}"
CFG="$PWD/configs/swin_tiny_patch4_window7_224_lite.yaml"
common=(--root_path "$DATA_ROOT" --cfg "$CFG" --img_size 256 --source_patch_size 1024
        --enable_highres_structure_stream --highres_structure_channels 64
        --highres_structure_fuse_stages stage23 --highres_structure_fusion_mode stage23
        --structure_profile stage23_boundary_0626 --stage_skeleton_mode direct
        --enable_h3_surface_fusion --remove_stage2_pre_topology_source
        --enable_global_topology --global_topology_max_nodes 32 --global_topology_heads 4
        --global_topology_alpha_max 0.05 --global_topology_mode supervised_anchors)
case "$MODE" in
  train|resume)
    if [[ "$MODE" == train && -e "$RUN_DIR" ]]; then
      echo "RUN_DIR already exists; choose another RUN or use resume: $RUN_DIR" >&2
      exit 1
    fi
    extra=()
    if [[ "$MODE" == resume ]]; then
      [[ -s "$RUN_DIR/last.pth" ]] || { echo "Missing $RUN_DIR/last.pth" >&2; exit 1; }
      extra=(--resume "$RUN_DIR/last.pth")
    fi
    # train_image creates the experiment folder itself, avoiding an unwanted _2 suffix.
    mkdir -p "$OUTPUT_ROOT"
    log="$OUTPUT_ROOT/${RUN}_${MODE}.log"
    python -u train_image.py "${common[@]}" --output_dir "$OUTPUT_ROOT" --run_name "$RUN" \
      --pretrain_ckpt "$PRETRAIN" --direct_resize_train --batch_size 4 --max_epochs "$MAX_EPOCHS" \
      --warmup_epochs 10 --val_interval 5 --num_workers "$WORKERS" \
      --base_lr 5e-4 --min_lr 1e-5 --pretrained_lr 5e-5 --pretrained_min_lr 5e-6 \
      --new_lr 2e-4 --new_min_lr 1e-5 --threshold 0.2 --skeleton_threshold 0.5 \
      --stage2_skeleton_weight 0.008 --stage3_skeleton_weight 0.012 --skeleton_pos_weight 2.0 \
      --stage_connectivity_factor 3.0 --stage_direction_factor 0.2 \
      --masked_connectivity_center_experiment --connectivity_pos_weight 2.0 \
      --connectivity_focal_gamma 1.5 --edge_contrastive_margin 0.1 \
      --directional_pos_weight_cardinal 1.0 --directional_pos_weight_diagonal 2.5 \
      --road_attention_weight 0.0 --highres_structure_skeleton_weight 0.0 --final_skeleton_weight 0.0 \
      --surface_focal_gamma 1.0 --amp_dtype none --seed 1234 --deterministic 1 --no-use_ema \
      --anchor_neighbours 4 --anchor_max_distance 64 --anchor_samples 8 --anchor_corridor_offset 2 \
      --anchor_prior_warmup_epochs 10 --anchor_prior_ramp_epochs 5 \
      --anchor_connection_loss_weight 0.1 --anchor_snap_radius 3 --anchor_max_geodesic 96 \
      --anchor_max_detour 2 --anchor_cache_dir "$CACHE_DIR" "${extra[@]}" 2>&1 | tee -a "$log"
    ;;
  sweep)
    [[ -s "$RUN_DIR/best.pth" ]] || { echo "Missing $RUN_DIR/best.pth" >&2; exit 1; }
    python -u threshold_sweep_test.py "${common[@]}" --model_path "$RUN_DIR/best.pth" \
      --split val --batch_size 4 --num_workers "$WORKERS" --output_json "$RUN_DIR/threshold_val.json" \
      2>&1 | tee "$RUN_DIR/threshold_val.log"
    ;;
  test)
    [[ -s "$RUN_DIR/best.pth" ]] || { echo "Missing $RUN_DIR/best.pth" >&2; exit 1; }
    if [[ -z "${BEST_THRESHOLD:-}" ]]; then
      [[ -s "$RUN_DIR/threshold_val.json" ]] || { echo 'Run sweep on val first, or set BEST_THRESHOLD' >&2; exit 1; }
      BEST_THRESHOLD=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_threshold"])' "$RUN_DIR/threshold_val.json")
    fi
    python -u test_image.py "${common[@]}" --model_path "$RUN_DIR/best.pth" \
      --threshold "$BEST_THRESHOLD" --batch_size 1 --num_workers "$WORKERS" \
      --output_dir "$RUN_DIR/test_thr_$BEST_THRESHOLD" 2>&1 | tee "$RUN_DIR/test_thr_${BEST_THRESHOLD}.log"
    ;;
  *) echo 'Usage: bash scripts/run_h0_anchor_topology_server.sh cache|train|resume|sweep|test' >&2; exit 1 ;;
esac
