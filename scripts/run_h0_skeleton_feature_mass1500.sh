#!/usr/bin/env bash
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=${DATA:-/home/gjj/Swin-Unet-main/Massachusetts/tiff}
RESULTS=${RESULTS:-/home/gjj/SwinNet-roadbias/model_out}
PRETRAIN=${PRETRAIN:-/home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth}
SEED=${SEED:-1234}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OPENCV_LOG_LEVEL=ERROR
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODE=${1:-}
if [[ "$MODE" != train && "$MODE" != resume && "$MODE" != sweep && "$MODE" != test && "$MODE" != topology ]]; then
  echo "Usage: bash scripts/run_h0_skeleton_feature_mass1500.sh train|resume|sweep|test|topology" >&2
  exit 2
fi
cd "$REPO"
mkdir -p "$RESULTS"
LATEST="$RESULTS/latest_h0_skeleton_feature_mass1500_run.txt"
if [[ "$MODE" == train ]]; then
  test -s "$PRETRAIN" || { echo "Missing pretrained checkpoint: $PRETRAIN" >&2; exit 1; }
  RUN=${RUN:-h0_skeleton_feature_mass1500_random512_fp32_noema80_b2_acc1_s${SEED}_$(date +%Y%m%d_%H%M%S)}
  test ! -e "$RESULTS/$RUN" || { echo "Existing run: $RUN; use resume or a new RUN" >&2; exit 1; }
  python -u scripts/check_mass_layout.py --root_path "$DATA" --source_size 1500
  printf '%s\n' "$RUN" > "$LATEST"
else
  if [[ -z "${RUN:-}" ]]; then
    test -s "$LATEST" || { echo "Set RUN to the completed training run name" >&2; exit 1; }
    RUN=$(<"$LATEST")
  fi
fi
RUN_DIR="$RESULTS/$RUN"
CFG="$REPO/configs/swin_tiny_patch4_window7_224_lite.yaml"
EVAL="$REPO/evaluation/road_comparison"
echo "RUN_DIR=$RUN_DIR; GPU=$CUDA_VISIBLE_DEVICES; FP32; no EMA; physical batch 2; accumulation 1"

if [[ "$MODE" == train || "$MODE" == resume ]]; then
  resume_args=()
  if [[ "$MODE" == resume ]]; then
    test -s "$RUN_DIR/last.pth" || { echo "Missing $RUN_DIR/last.pth" >&2; exit 1; }
    python "$EVAL/build_skeleton_feature_manifest.py" --checkpoint "$RUN_DIR/last.pth" \
      --data_root "$DATA" --predictions_root "$RUN_DIR" --output "$RUN_DIR/resume_check.json" --check_only
    resume_args+=(--resume "$RUN_DIR/last.pth")
  fi
  python -u train_image.py \
    --root_path "$DATA" --output_dir "$RESULTS" --run_name "$RUN" \
    "${resume_args[@]}" --cfg "$CFG" --pretrain_ckpt "$PRETRAIN" \
    --img_size 512 --source_patch_size 1500 --random_crop_train --random_crops_per_image 1 \
    --overlap_stride 256 --batch_size 2 --accumulation_steps 1 --max_epochs 80 \
    --warmup_epochs 10 --val_interval 5 --num_workers 4 --trend_val_every_steps 0 \
    --amp_dtype none --no-use_ema --seed "$SEED" --deterministic 1 \
    --base_lr 5e-4 --min_lr 1e-5 --pretrained_lr 5e-5 --pretrained_min_lr 5e-6 \
    --new_lr 2e-4 --new_min_lr 1e-5 --threshold 0.2 --skeleton_threshold 0.5 \
    --structure_profile stage23_boundary_0626 --stage_skeleton_mode direct \
    --enable_highres_structure_stream --highres_structure_channels 64 \
    --highres_structure_fuse_stages stage23 --highres_structure_fusion_mode stage23 \
    --remove_stage2_pre_topology_source \
    --stage2_skeleton_weight 0.008 --stage3_skeleton_weight 0.012 --skeleton_pos_weight 2.0 \
    --stage_skeleton_loss_factor 1.0 --stage_skeleton_only_loss_factor 0.5 \
    --stage2_skeleton_gradient_ratio 0.5 --stage3_skeleton_gradient_ratio 0.5 \
    --stage3_gate_topology_gradient_ratio 0.0 --highres_structure_skeleton_weight 0.01 \
    --stage_direction_factor 0.2 --stage_connectivity_factor 3.0 --road_attention_weight 0.0 \
    --masked_connectivity_center_experiment --connectivity_pos_weight 2.0 \
    --connectivity_focal_gamma 1.5 --edge_contrastive_margin 0.1 \
    --directional_pos_weight_cardinal 1.0 --directional_pos_weight_diagonal 2.5 \
    --surface_focal_gamma 1.0 \
    2>&1 | tee -a "$RESULTS/${RUN}.log"
  echo "BEST=$RUN_DIR/best.pth"
  exit 0
fi

CKPT="$RUN_DIR/best.pth"
test -s "$CKPT" || { echo "Missing $CKPT" >&2; exit 1; }
AUDIT_POINTER="$RUN_DIR/latest_unified_eval.txt"
if [[ "$MODE" == sweep ]]; then
  AUDIT="$RUN_DIR/unified_eval_$(date +%Y%m%d_%H%M%S)"
  test ! -e "$AUDIT" || { echo "Existing audit: $AUDIT" >&2; exit 1; }
  python "$EVAL/build_skeleton_feature_manifest.py" --checkpoint "$CKPT" \
    --data_root "$DATA" --predictions_root "$AUDIT/predictions" --output "$AUDIT/comparison.json" --check_only
  mkdir -p "$AUDIT"
  python -u test_image.py --root_path "$DATA" --model_path "$CKPT" --cfg "$CFG" \
    --split val --img_size 512 --source_patch_size 1500 --overlap_infer --overlap_stride 256 \
    --output_dir "$AUDIT/predictions/val" \
    --export_thresholds 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 \
                        0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 \
    2>&1 | tee "$AUDIT/val_inference.log"
  python "$EVAL/build_skeleton_feature_manifest.py" --checkpoint "$CKPT" \
    --data_root "$DATA" --predictions_root "$AUDIT/predictions" --output "$AUDIT/comparison.json"
  python -u "$EVAL/compare_predictions.py" --manifest "$AUDIT/comparison.json" \
    --split val --select_threshold_only --output_dir "$AUDIT/val" --progress_every 1 \
    2>&1 | tee "$AUDIT/val_metrics.log"
  printf '%s\n' "$AUDIT" > "$AUDIT_POINTER"
  echo "VAL_SELECTION=$AUDIT/val/val_selection.json"
  exit 0
fi

test -s "$AUDIT_POINTER" || { echo "Run sweep first" >&2; exit 1; }
AUDIT=$(<"$AUDIT_POINTER")
MANIFEST="$AUDIT/comparison.json"
SELECTION="$AUDIT/val/val_selection.json"
test -s "$SELECTION" || { echo "Missing $SELECTION" >&2; exit 1; }
THRESHOLD=$(python -c 'import json,sys; print(next(iter(json.load(open(sys.argv[1]))["models"].values()))["threshold"])' "$SELECTION")
echo "Validation-selected threshold=$THRESHOLD"
if [[ "$MODE" == test ]]; then
  python "$EVAL/build_skeleton_feature_manifest.py" --checkpoint "$CKPT" \
    --data_root "$DATA" --predictions_root "$AUDIT/predictions" --output "$MANIFEST"
  python -u test_image.py --root_path "$DATA" --model_path "$CKPT" --cfg "$CFG" \
    --split test --img_size 512 --source_patch_size 1500 --overlap_infer --overlap_stride 256 \
    --output_dir "$AUDIT/predictions/test" --threshold "$THRESHOLD" --export_thresholds "$THRESHOLD" \
    2>&1 | tee "$AUDIT/test_inference.log"
  python "$EVAL/build_skeleton_feature_manifest.py" --checkpoint "$CKPT" \
    --data_root "$DATA" --predictions_root "$AUDIT/predictions" --output "$MANIFEST" --selection "$SELECTION"
fi
METRICS_OUT="$AUDIT/test"
if [[ "$MODE" == topology ]]; then
  METRICS_OUT="$AUDIT/test_recomputed_$(date +%Y%m%d_%H%M%S)"
fi
python -u "$EVAL/compare_predictions.py" --manifest "$MANIFEST" --split test \
  --selection "$SELECTION" --output_dir "$METRICS_OUT" --progress_every 1 \
  2>&1 | tee "$AUDIT/$(basename "$METRICS_OUT")_metrics.log"
echo "TEST_TOPOLOGY_CSV=$METRICS_OUT/test_comparison.csv"
cat "$METRICS_OUT/test_comparison.csv"
