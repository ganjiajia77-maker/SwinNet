#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=${DATA:-/home/gjj/Swin-Unet-main/Massachusetts/tiff}
RESULTS=${RESULTS:-/home/gjj/SwinNet-roadbias/model_out}
PRETRAIN=${PRETRAIN:-/home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth}
PAIR_NAME=${PAIR_NAME:-h0_native_relation_80e_s1234}
SEED=${SEED:-1234}
MODE=${1:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export OPENCV_LOG_LEVEL=ERROR
cd "$REPO"
test -f "$PRETRAIN" || { echo "Missing ImageNet checkpoint: $PRETRAIN" >&2; exit 1; }
test -d "$DATA/train" || { echo "Missing training data: $DATA" >&2; exit 1; }
mkdir -p "$RESULTS"
common=(--root_path "$DATA" --output_dir "$RESULTS" --cfg "$REPO/configs/swin_tiny_patch4_window7_224_lite.yaml"
  --pretrain_ckpt "$PRETRAIN" --pretrained_lr 5e-5 --pretrained_min_lr 5e-6
  --new_lr 2e-4 --new_min_lr 1e-5 --warmup_epochs 10 --amp_dtype none
  --max_epochs 80 --val_interval 5 --batch_size 2 --accumulation_steps 1 --num_workers 4
  --disable_persistent_workers --img_size 512 --source_patch_size 1500 --overlap_stride 256
  --random_crop_train --random_crops_per_image 1 --no-use_ema --seed "$SEED" --threshold 0.2
  --structure_profile stage23_boundary_0626 --stage_skeleton_mode direct
  --enable_highres_structure_stream --enable_h3_surface_fusion --remove_stage2_pre_topology_source
  --highres_structure_fuse_stages stage23 --highres_structure_fusion_mode stage23
  --enable_global_topology --global_topology_max_nodes 32 --global_topology_heads 4 --global_topology_alpha_max 0.05
  --stage2_skeleton_weight 0.008 --stage3_skeleton_weight 0.012
  --stage2_skeleton_gradient_ratio 0.5 --stage3_skeleton_gradient_ratio 0.5
  --stage3_gate_topology_gradient_ratio 0.0 --skeleton_pos_weight 2.0
  --highres_structure_skeleton_weight 0.0 --stage_direction_factor 0.2 --stage_connectivity_factor 3.0
  --road_attention_weight 0.0 --masked_connectivity_center_experiment
  --connectivity_pos_weight 2.0 --connectivity_focal_gamma 1.5 --edge_contrastive_margin 0.1
  --directional_pos_weight_cardinal 1.0 --directional_pos_weight_diagonal 2.5 --surface_focal_gamma 1.0
  --record_initial_state)
case "$MODE" in
  local_cs)
    : "${LOCAL_CS_INIT:?Set an explicit original native B checkpoint}"
    : "${LOCAL_CS_RUN:?Set an independent run name}"
    test -s "$LOCAL_CS_INIT"
    test ! -e "$RESULTS/$LOCAL_CS_RUN" || { echo 'Output exists; choose a new run name' >&2; exit 1; }
    python -u train_image.py "${common[@]}" --native_relation_experiment \
      --run_name "$LOCAL_CS_RUN" --local_cs_initial_checkpoint "$LOCAL_CS_INIT" \
      --local_cs_mode "${LOCAL_CS_MODE:-c_guided}" --lambda_local_cs "${LAMBDA_LOCAL_CS:-0.05}" \
      --finetune_epochs "${SHORT_EPOCHS:-1}" --max_train_batches "${SHORT_BATCHES:-100}" \
      --local_cs_audit_interval "${LOCAL_CS_AUDIT_INTERVAL:-50}"
    ;;
  prepare)
    for run in "${PAIR_NAME}_preflight_A" "${PAIR_NAME}_preflight_B"; do
      test ! -e "$RESULTS/$run" || { echo "Existing preflight: $RESULTS/$run" >&2; exit 1; }
    done
    python -u train_image.py "${common[@]}" --run_name "${PAIR_NAME}_preflight_A" --initial_manifest_only
    python -u train_image.py "${common[@]}" --run_name "${PAIR_NAME}_preflight_B" --native_relation_experiment --native_smoke_only
    python -u scripts/compare_native_initial_states.py \
      "$RESULTS/${PAIR_NAME}_preflight_A/initial_state.json" "$RESULTS/${PAIR_NAME}_preflight_B/initial_state.json" \
      --output "$RESULTS/${PAIR_NAME}_initial_comparison.json"
    ;;
  A|B)
    test -s "$RESULTS/${PAIR_NAME}_initial_comparison.json" || { echo 'Run prepare first' >&2; exit 1; }
    test -s "$RESULTS/${PAIR_NAME}_preflight_B/native_smoke.json" || { echo 'Native 512-batch smoke missing' >&2; exit 1; }
    python -c 'import json,sys; a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); assert a["same_seed"] and a["shared_tensors_checked"]>0 and not a["mismatches"] and b["passed"], "Preflight failed"' \
      "$RESULTS/${PAIR_NAME}_initial_comparison.json" "$RESULTS/${PAIR_NAME}_preflight_B/native_smoke.json"
    run="${PAIR_NAME}_${MODE}";test ! -e "$RESULTS/$run" || { echo "Existing run: $RESULTS/$run" >&2; exit 1; }
    extra=();if [[ "$MODE" == B ]]; then extra+=(--native_relation_experiment); fi
    python -u train_image.py "${common[@]}" --run_name "$run" "${extra[@]}" \
      --verify_initial_state "$RESULTS/${PAIR_NAME}_preflight_${MODE}/initial_state.json"
    ;;
  resume_A|resume_B)
    variant=${MODE#resume_};run="${PAIR_NAME}_${variant}"
    test -s "$RESULTS/$run/last.pth" || { echo "Missing resume checkpoint: $RESULTS/$run/last.pth" >&2; exit 1; }
    extra=();if [[ "$variant" == B ]]; then extra+=(--native_relation_experiment); fi
    python -u train_image.py "${common[@]}" --run_name "$run" "${extra[@]}" --resume "$RESULTS/$run/last.pth"
    ;;
  val)
    python -u scripts/evaluate_native_h0_pair.py --baseline "$RESULTS/${PAIR_NAME}_A/best.pth" \
      --native "$RESULTS/${PAIR_NAME}_B/best.pth" --root_path "$DATA" \
      --output_dir "$RESULTS/${PAIR_NAME}_paired_val" --device cuda
    ;;
  *) echo 'Usage: bash scripts/run_native_h0_pair.sh prepare|A|B|resume_A|resume_B|val' >&2; exit 2 ;;
esac
