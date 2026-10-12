#!/usr/bin/env bash
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=${DATA:-/home/gjj/Swin-Unet-main/Massachusetts/tiff}
RESULTS=${RESULTS:-/home/gjj/SwinNet-roadbias/model_out}
PRETRAIN=${PRETRAIN:-/home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth}
SEED=${SEED:-1234}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export OPENCV_LOG_LEVEL=ERROR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODE=${1:-}
if [[ "$MODE" != train && "$MODE" != sweep && "$MODE" != test && "$MODE" != topology ]]; then
  echo "Usage: bash scripts/run_h0_road_prior_mass.sh train|sweep|test|topology" >&2
  exit 2
fi

cd "$REPO"
mkdir -p "$RESULTS"
LATEST="$RESULTS/latest_mass_h0_no_encoder_road_prior_run.txt"
if [[ "$MODE" == train ]]; then
  test -s "$PRETRAIN" || { echo "Missing pretrained checkpoint: $PRETRAIN" >&2; exit 1; }
  python -u scripts/check_mass_layout.py --root_path "$DATA" --source_size 1500
  RUN=${RUN:-h0_mass1500_no_encoder_road_prior_fp32_noema80_b2_acc1_s${SEED}_$(date +%Y%m%d_%H%M%S)}
  test ! -e "$RESULTS/$RUN" || { echo "Existing run: $RUN" >&2; exit 1; }
  printf '%s\n' "$RUN" > "$LATEST"
else
  if [[ -z "${RUN:-}" ]]; then
    test -s "$LATEST" || { echo "Set RUN to the training run name" >&2; exit 1; }
    RUN=$(<"$LATEST")
  fi
fi
RUN_DIR="$RESULTS/$RUN"
CFG="$REPO/configs/swin_tiny_patch4_window7_224_lite.yaml"
EVAL="$REPO/evaluation/road_comparison"

if [[ "$MODE" == train ]]; then
  echo "RUN_DIR=$RUN_DIR; GPU=$CUDA_VISIBLE_DEVICES; FP32; no EMA; physical batch 2; accumulation 1"
  python -u train_image.py \
    --root_path "$DATA" --output_dir "$RESULTS" --run_name "$RUN" \
    --cfg "$CFG" --pretrain_ckpt "$PRETRAIN" \
    --img_size 512 --source_patch_size 1500 --overlap_stride 256 \
    --random_crop_train --random_crops_per_image 1 --max_epochs 80 \
    --batch_size 2 --accumulation_steps 1 --val_interval 5 --num_workers 4 \
    --seed "$SEED" --amp_dtype none --no-use_ema --trend_val_every_steps 0 \
    --threshold 0.5 --base_lr 5e-4 \
    --pretrained_lr 5e-5 --pretrained_min_lr 5e-6 \
    --new_lr 2e-4 --new_min_lr 1e-5 --warmup_epochs 10 \
    --structure_profile stage23_boundary_0626 --stage_skeleton_mode direct \
    --enable_highres_structure_stream --highres_structure_channels 64 \
    --enable_h3_surface_fusion --remove_stage2_pre_topology_source \
    --highres_structure_fuse_stages stage23 --highres_structure_fusion_mode stage23 \
    --enable_global_topology --global_topology_max_nodes 32 \
    --global_topology_heads 4 --global_topology_alpha_max 0.05 \
    --no-enable_encoder_road_prior \
    --stage2_skeleton_weight 0.008 --stage3_skeleton_weight 0.012 \
    --stage_skeleton_loss_factor 1.0 --stage_skeleton_only_loss_factor 1.0 \
    --stage2_skeleton_gradient_ratio 0.5 --stage3_skeleton_gradient_ratio 0.5 \
    --stage3_gate_topology_gradient_ratio 0.0 --skeleton_pos_weight 2.0 \
    --highres_structure_skeleton_weight 0.0 --stage_direction_factor 0.2 \
    --stage_connectivity_factor 3.0 --road_attention_weight 0.0 \
    --masked_connectivity_center_experiment --connectivity_pos_weight 2.0 \
    --connectivity_focal_gamma 1.5 --edge_contrastive_margin 0.1 \
    --directional_pos_weight_cardinal 1.0 --directional_pos_weight_diagonal 2.5 \
    --surface_focal_gamma 1.0 \
    2>&1 | tee "$RESULTS/${RUN}.log"
  echo "BEST=$RUN_DIR/best.pth"
  exit 0
fi

CKPT="$RUN_DIR/best.pth"
test -s "$CKPT" || { echo "Missing $CKPT" >&2; exit 1; }
AUDIT_POINTER="$RUN_DIR/latest_road_prior_audit.txt"
if [[ "$MODE" == sweep ]]; then
  AUDIT="$RUN_DIR/road_prior_audit_$(date +%Y%m%d_%H%M%S)"
  test ! -e "$AUDIT" || { echo "Existing audit: $AUDIT" >&2; exit 1; }
  mkdir -p "$AUDIT"
  PREDICTIONS="$AUDIT/predictions"
  MANIFEST="$AUDIT/comparison.json"
  python "$EVAL/build_h0_manifest.py" --checkpoint "$CKPT" --data_root "$DATA" \
    --predictions_root "$PREDICTIONS" --output "$MANIFEST" \
    --expected_accumulation_steps 1 --check_only
  python -u test_image.py --root_path "$DATA" --model_path "$CKPT" \
    --cfg "$CFG" --split val --img_size 512 --source_patch_size 1500 \
    --overlap_infer --output_dir "$PREDICTIONS/val" \
    --export_thresholds 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 \
                        0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 \
    2>&1 | tee "$AUDIT/val_inference.log"
  python "$EVAL/build_h0_manifest.py" --checkpoint "$CKPT" --data_root "$DATA" \
    --predictions_root "$PREDICTIONS" --output "$MANIFEST" \
    --expected_accumulation_steps 1
  python -u "$EVAL/compare_predictions.py" --manifest "$MANIFEST" \
    --split val --select_threshold_only --output_dir "$AUDIT/val" \
    --progress_every 10 2>&1 | tee "$AUDIT/val_metrics.log"
  printf '%s\n' "$AUDIT" > "$AUDIT_POINTER"
  echo "SWEEP_CSV=$AUDIT/val/val_threshold_scores.csv"
  echo "SELECTION=$AUDIT/val/val_selection.json"
  exit 0
fi

test -s "$AUDIT_POINTER" || { echo "Run sweep first" >&2; exit 1; }
AUDIT=$(<"$AUDIT_POINTER")
PREDICTIONS="$AUDIT/predictions"
MANIFEST="$AUDIT/comparison.json"
SELECTION="$AUDIT/val/val_selection.json"
test -s "$SELECTION" || { echo "Missing validation selection: $SELECTION" >&2; exit 1; }
THRESHOLD=$(python -c 'import json,sys; print(next(iter(json.load(open(sys.argv[1]))["models"].values()))["threshold"])' "$SELECTION")
echo "Validation-selected surface threshold: $THRESHOLD"

if [[ "$MODE" == test ]]; then
  python -u test_image.py --root_path "$DATA" --model_path "$CKPT" \
    --cfg "$CFG" --split test --img_size 512 --source_patch_size 1500 \
    --overlap_infer --output_dir "$PREDICTIONS/test" \
    --threshold "$THRESHOLD" --export_thresholds "$THRESHOLD" \
    2>&1 | tee "$AUDIT/test_inference.log"
  python "$EVAL/build_h0_manifest.py" --checkpoint "$CKPT" --data_root "$DATA" \
    --predictions_root "$PREDICTIONS" --output "$MANIFEST" \
    --expected_accumulation_steps 1 --selection "$SELECTION"
  echo "TEST_PREDICTIONS=$PREDICTIONS/test/threshold_$(printf '%.2f' "$THRESHOLD")/surface"
  exit 0
fi

test -s "$MANIFEST" || { echo "Run test first to generate a test manifest" >&2; exit 1; }
python -u "$EVAL/compare_predictions.py" --manifest "$MANIFEST" \
  --split test --selection "$SELECTION" --output_dir "$AUDIT/test" \
  --progress_every 10 2>&1 | tee "$AUDIT/test_metrics.log"
echo "TOPOLOGY_CSV=$AUDIT/test/test_comparison.csv"
echo "PER_IMAGE_CSV=$AUDIT/test"
