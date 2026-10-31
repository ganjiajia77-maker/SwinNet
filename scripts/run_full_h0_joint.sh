#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
VARIANT=${1:?Usage: run_full_h0_joint.sh A|B [all|smoke|val|compare]}
MODE=${2:-all}
[[ "$VARIANT" == A || "$VARIANT" == B ]] || { echo 'Choose A or B'; exit 2; }
GPU=${GPU:-1}
export CUDA_VISIBLE_DEVICES="$GPU"
DATA=${DATA:-/home/gjj/Swin-Unet-main/Massachusetts/tiff}
PAIR_DIR=${PAIR_DIR:-/home/gjj/SwinNet-roadbias/model_out/full_h0_joint_pair_v1}
EPOCHS=${EPOCHS:-80}
mkdir -p "$PAIR_DIR/$VARIANT"
exec 9>"$PAIR_DIR/$VARIANT/.run.lock"
flock -n 9 || { echo "Variant $VARIANT already running"; exit 1; }
if [[ "$MODE" == compare ]]; then
  python -u compare_full_h0_joint.py --pair_dir "$PAIR_DIR"
  exit
fi
args=(--variant "$VARIANT" --pair_dir "$PAIR_DIR" --root_path "$DATA" --device cuda --epochs "$EPOCHS")
if [[ -n "${H0_CKPT:-}" ]]; then args+=(--h0_checkpoint "$H0_CKPT"); fi
if [[ -n "${REPAIR_RUN:-}" ]]; then args+=(--repair_run "$REPAIR_RUN"); fi
# Only the initial shared-reference pass is serialised; A/B train concurrently.
(
  flock 8
  python -u run_full_h0_joint.py prepare "${args[@]}"
) 8>"$PAIR_DIR/.reference.lock"
python -u run_full_h0_joint.py "$MODE" "${args[@]}" 2>&1 | tee -a "$PAIR_DIR/$VARIANT/console.log"
if [[ "$MODE" == all ]]; then
  (
    flock 8
    python -u compare_full_h0_joint.py --pair_dir "$PAIR_DIR"
  ) 8>"$PAIR_DIR/.compare.lock"
fi
