#!/usr/bin/env bash
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=${DATA:-/home/gjj/Swin-Unet-main/Massachusetts/tiff}
RESULTS=${RESULTS:-/home/gjj/SwinNet-roadbias/model_out}
PRETRAIN=${PRETRAIN:-/home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth}
VARIANT=${VARIANT:-a}
SEED=${SEED:-1234}
USE_EMA=${USE_EMA:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export OPENCV_LOG_LEVEL=ERROR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"

MODE=${1:-}
if [[ "$VARIANT" != a && "$VARIANT" != b && "$VARIANT" != full ]]; then
  echo "VARIANT must be a, b or full" >&2
  exit 2
fi
if [[ "$USE_EMA" != 0 && "$USE_EMA" != 1 ]]; then
  echo "USE_EMA must be 0 or 1" >&2
  exit 2
fi
if [[ "$MODE" != train && "$MODE" != resume && "$MODE" != val && "$MODE" != test && "$MODE" != topology ]]; then
  echo "Usage: VARIANT=a|b|full bash scripts/run_h0_mass1500.sh train|resume|val|test|topology" >&2
  exit 2
fi
mkdir -p "$RESULTS"
LATEST="$RESULTS/latest_mass_h0_${VARIANT}_run.txt"
RUN=${RUN:-}
if [[ "$MODE" == train ]]; then
  test -f "$PRETRAIN" || { echo "Missing pretrained checkpoint: $PRETRAIN" >&2; exit 1; }
  python -u scripts/check_mass_layout.py --root_path "$DATA" --source_size 1500
  ema_label=noema
  if [[ "$USE_EMA" == 1 ]]; then ema_label=ema; fi
  RUN=${RUN:-h0_mass1500_${VARIANT}_random512_fp32_${ema_label}80_s${SEED}_$(date +%Y%m%d_%H%M%S)}
  test ! -e "$RESULTS/$RUN" || { echo "Existing run: $RUN" >&2; exit 1; }
  printf '%s\n' "$RUN" > "$LATEST"
else
  RUN=${RUN:-$(cat "$LATEST")}
fi
RUN_DIR="$RESULTS/$RUN"

if [[ "$MODE" == train || "$MODE" == resume ]]; then
  extra=()
  if [[ "$USE_EMA" == 1 ]]; then
    extra+=(--use_ema)
  else
    extra+=(--no-use_ema)
  fi
  if [[ "$VARIANT" == a ]]; then
    extra+=(--ablate_h0_stream)
  elif [[ "$VARIANT" == b ]]; then
    extra+=(--enable_highres_structure_stream --static_h0_stage_fusion)
  else
    extra+=(--enable_highres_structure_stream)
  fi
  if [[ "$MODE" == resume ]]; then
    test -s "$RUN_DIR/last.pth" || { echo "Missing $RUN_DIR/last.pth" >&2; exit 1; }
    saved_ema=$(python -c 'import sys,torch; c=torch.load(sys.argv[1],map_location="cpu",weights_only=False); print(int(c["args"]["use_ema"]))' "$RUN_DIR/last.pth")
    if [[ "$saved_ema" != "$USE_EMA" ]]; then
      echo "Resume checkpoint uses EMA=$saved_ema; set USE_EMA=$saved_ema" >&2
      exit 1
    fi
    extra+=(--resume "$RUN_DIR/last.pth")
  fi
  python -u train_image.py \
    --root_path "$DATA" --output_dir "$RESULTS" --run_name "$RUN" \
    "${extra[@]}" \
    --cfg "$REPO/configs/swin_tiny_patch4_window7_224_lite.yaml" \
    --pretrain_ckpt "$PRETRAIN" \
    --pretrained_lr 5e-5 --pretrained_min_lr 5e-6 \
    --new_lr 2e-4 --new_min_lr 1e-5 --warmup_epochs 10 \
    --amp_dtype none --max_epochs 80 --val_interval 5 \
    --batch_size 2 --accumulation_steps 2 --num_workers 4 \
    --img_size 512 --source_patch_size 1500 --overlap_stride 256 \
    --random_crop_train --random_crops_per_image 1 \
    --ema_decay 0.999 --seed "$SEED" --threshold 0.2 \
    --structure_profile stage23_boundary_0626 --stage_skeleton_mode direct \
    --enable_h3_surface_fusion --remove_stage2_pre_topology_source \
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
    2>&1 | tee -a "$RESULTS/${RUN}.log"
  exit 0
fi

CKPT="$RUN_DIR/best.pth"
test -s "$CKPT" || { echo "Missing $CKPT" >&2; exit 1; }
eval_weights=()
if [[ "$(python -c 'import sys,torch; c=torch.load(sys.argv[1],map_location="cpu",weights_only=False); print(int(c["args"]["use_ema"]))' "$CKPT")" == 1 ]]; then
  eval_weights+=(--require_ema)
fi
EVAL="$REPO/evaluation/road_comparison"
if [[ "$MODE" == val ]]; then
  AUDIT="$RUN_DIR/unified_$(date +%Y%m%d_%H%M%S)"
  test ! -e "$AUDIT" || { echo "Existing audit directory: $AUDIT" >&2; exit 1; }
  mkdir -p "$AUDIT"
  python "$EVAL/build_h0_manifest.py" --checkpoint "$CKPT" --data_root "$DATA" \
    --predictions_root "$AUDIT/predictions" --output "$AUDIT/comparison.json" --check_only
  printf '%s\n' "$AUDIT" > "$RUN_DIR/latest_unified_audit.txt"
  python -u test_image.py --root_path "$DATA" --model_path "$CKPT" \
    --cfg "$REPO/configs/swin_tiny_patch4_window7_224_lite.yaml" \
    --split val --img_size 512 --source_patch_size 1500 --overlap_infer "${eval_weights[@]}" \
    --output_dir "$AUDIT/predictions/val" \
    --export_thresholds 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 \
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
THRESHOLD=$(python -c 'import json,sys; print(next(iter(json.load(open(sys.argv[1]))["models"].values()))["threshold"])' "$SELECTION")
echo "Validation-selected threshold: $THRESHOLD"
if [[ "$MODE" == test ]]; then
  python -u test_image.py --root_path "$DATA" --model_path "$CKPT" \
    --cfg "$REPO/configs/swin_tiny_patch4_window7_224_lite.yaml" \
    --split test --img_size 512 --source_patch_size 1500 --overlap_infer "${eval_weights[@]}" \
    --output_dir "$AUDIT/predictions/test" --threshold "$THRESHOLD" --export_thresholds "$THRESHOLD" \
    2>&1 | tee "$AUDIT/test_inference.log"
  python "$EVAL/build_h0_manifest.py" --checkpoint "$CKPT" --data_root "$DATA" \
    --predictions_root "$AUDIT/predictions" --output "$AUDIT/comparison.json" --selection "$SELECTION"
fi
python -u "$EVAL/compare_predictions.py" --manifest "$AUDIT/comparison.json" \
  --split test --selection "$SELECTION" --output_dir "$AUDIT/test" --progress_every 10 \
  2>&1 | tee "$AUDIT/test_metrics.log"
cat "$AUDIT/test/test_comparison.csv"
