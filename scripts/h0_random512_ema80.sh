#!/usr/bin/env bash
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=${DATA:-/home/gjj/Swin-Unet-main/data1}
RESULTS=${RESULTS:-/home/gjj/SwinNet-h0-results}
PRETRAIN=${PRETRAIN:-/home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OPENCV_LOG_LEVEL=ERROR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
MODE=${1:-}
mkdir -p "$RESULTS"

if [[ "$MODE" == train ]]; then
  test -f "$PRETRAIN" || { echo "Missing pretrained checkpoint: $PRETRAIN"; exit 1; }
  RUN=${RUN:-h0_599a410_data1_random512_fp32_ema0999_80e_$(date +%Y%m%d_%H%M%S)}
  test ! -e "$RESULTS/$RUN" || { echo "Existing run: $RUN; choose a new RUN"; exit 1; }
  printf '%s\n' "$RUN" > "$RESULTS/latest_h0_run.txt"
  python -u train_image.py \
    --root_path "$DATA" --output_dir "$RESULTS" --run_name "$RUN" \
    --cfg "$REPO/configs/swin_tiny_patch4_window7_224_lite.yaml" \
    --pretrain_ckpt "$PRETRAIN" \
    --pretrained_lr 5e-5 --pretrained_min_lr 5e-6 \
    --new_lr 2e-4 --new_min_lr 1e-5 --warmup_epochs 10 \
    --amp_dtype none --max_epochs 80 --val_interval 5 \
    --batch_size 2 --accumulation_steps 2 --num_workers 4 \
    --img_size 512 --source_patch_size 1024 --overlap_stride 256 \
    --random_crop_train --random_crops_per_image 1 \
    --use_ema --ema_decay 0.999 --seed 1234 --threshold 0.2 \
    --structure_profile stage23_boundary_0626 --stage_skeleton_mode direct \
    --enable_highres_structure_stream --enable_h3_surface_fusion \
    --remove_stage2_pre_topology_source \
    --highres_structure_fuse_stages stage23 --highres_structure_fusion_mode stage23 \
    --enable_global_topology --global_topology_max_nodes 32 \
    --global_topology_heads 4 --global_topology_alpha_max 0.05 \
    --stage2_skeleton_weight 0.008 --stage3_skeleton_weight 0.012 \
    --stage2_skeleton_gradient_ratio 0.5 --stage3_skeleton_gradient_ratio 0.5 \
    --stage3_gate_topology_gradient_ratio 0.0 --skeleton_pos_weight 2.0 \
    --highres_structure_skeleton_weight 0.0 \
    --stage_direction_factor 0.2 --stage_connectivity_factor 3.0 \
    --road_attention_weight 0.0 --masked_connectivity_center_experiment \
    --connectivity_pos_weight 2.0 --connectivity_focal_gamma 1.5 \
    --edge_contrastive_margin 0.1 \
    --directional_pos_weight_cardinal 1.0 --directional_pos_weight_diagonal 2.5 \
    --surface_focal_gamma 1.0 \
    2>&1 | tee "$RESULTS/${RUN}.log"
  exit 0
fi

case "$MODE" in val|test|topology) ;; *) echo "Usage: bash scripts/h0_random512_ema80.sh train|val|test|topology"; exit 2 ;; esac
RUN=${RUN:-$(cat "$RESULTS/latest_h0_run.txt")}
RUN_DIR="$RESULTS/$RUN"
CKPT="$RUN_DIR/best.pth"
EVAL="$REPO/evaluation/road_comparison"
if [[ "$MODE" == val ]]; then
  AUDIT="$RUN_DIR/unified_$(date +%Y%m%d_%H%M%S)"
  test ! -e "$AUDIT" || { echo "Audit directory already exists"; exit 1; }
  mkdir -p "$AUDIT"
  python "$EVAL/build_h0_manifest.py" --checkpoint "$CKPT" --data_root "$DATA" \
    --predictions_root "$AUDIT/predictions" --output "$AUDIT/comparison.json" --check_only
  printf '%s\n' "$AUDIT" > "$RUN_DIR/latest_unified_audit.txt"
  python -u test_image.py --root_path "$DATA" --model_path "$CKPT" \
    --cfg "$REPO/configs/swin_tiny_patch4_window7_224_lite.yaml" \
    --split val --img_size 512 --source_patch_size 1024 --overlap_infer --require_ema \
    --output_dir "$AUDIT/predictions/val" \
    --export_thresholds 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 \
    2>&1 | tee "$AUDIT/val_inference.log"
  python "$EVAL/build_h0_manifest.py" --checkpoint "$CKPT" --data_root "$DATA" \
    --predictions_root "$AUDIT/predictions" --output "$AUDIT/comparison.json"
  python -u "$EVAL/compare_predictions.py" --manifest "$AUDIT/comparison.json" \
    --split val --output_dir "$AUDIT/val" --progress_every 10 \
    2>&1 | tee "$AUDIT/val_metrics.log"
  cat "$AUDIT/val/val_threshold_scores.csv"
  cat "$AUDIT/val/val_selection.json"
  exit 0
fi

AUDIT=$(cat "$RUN_DIR/latest_unified_audit.txt")
SELECTION="$AUDIT/val/val_selection.json"
THRESHOLD=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["models"]["h0_599a410_random512_fp32_ema80"]["threshold"])' "$SELECTION")
echo "Selected validation threshold: $THRESHOLD"
if [[ "$MODE" == test ]]; then
  python -u test_image.py --root_path "$DATA" --model_path "$CKPT" \
    --cfg "$REPO/configs/swin_tiny_patch4_window7_224_lite.yaml" \
    --split test --img_size 512 --source_patch_size 1024 --overlap_infer --require_ema \
    --output_dir "$AUDIT/predictions/test" --threshold "$THRESHOLD" --export_thresholds "$THRESHOLD" \
    2>&1 | tee "$AUDIT/test_inference.log"
  python "$EVAL/build_h0_manifest.py" --checkpoint "$CKPT" --data_root "$DATA" \
    --predictions_root "$AUDIT/predictions" --output "$AUDIT/comparison.json" --selection "$SELECTION"
fi
python -u "$EVAL/compare_predictions.py" --manifest "$AUDIT/comparison.json" \
  --split test --selection "$SELECTION" --output_dir "$AUDIT/test" --progress_every 10 \
  2>&1 | tee "$AUDIT/test_metrics.log"
cat "$AUDIT/test/test_comparison.csv"
