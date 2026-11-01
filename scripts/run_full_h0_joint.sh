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
B_RUN_NAME=${B_RUN_NAME:-B}
[[ "$B_RUN_NAME" =~ ^B(_[A-Za-z0-9_.-]+)?$ ]] || { echo 'Invalid B run name'; exit 2; }
RUN_SUBDIR=$VARIANT
if [[ "$VARIANT" == B ]]; then RUN_SUBDIR=$B_RUN_NAME; fi
mkdir -p "$PAIR_DIR/$RUN_SUBDIR"
exec 9>"$PAIR_DIR/$RUN_SUBDIR/.run.lock"
flock -n 9 || { echo "Variant $VARIANT already running"; exit 1; }
if [[ "$MODE" == compare ]]; then
  python -u compare_full_h0_joint.py --pair_dir "$PAIR_DIR" --b_run_name "$B_RUN_NAME"
  exit
fi
args=(--variant "$VARIANT" --pair_dir "$PAIR_DIR" --root_path "$DATA" --device cuda --epochs "$EPOCHS")
args+=(--b_run_name "$B_RUN_NAME")
if [[ -n "${TRANSPORT_RATE_INIT:-}" || -n "${PROPOSAL_RATE_INIT:-}" ]]; then
  [[ -n "${TRANSPORT_RATE_INIT:-}" && -n "${PROPOSAL_RATE_INIT:-}" ]] || { echo 'Set both initial budgets'; exit 2; }
  args+=(--transport_rate_init "$TRANSPORT_RATE_INIT" --proposal_rate_init "$PROPOSAL_RATE_INIT")
fi
if [[ -n "${H0_CKPT:-}" ]]; then args+=(--h0_checkpoint "$H0_CKPT"); fi
if [[ -n "${REPAIR_RUN:-}" ]]; then args+=(--repair_run "$REPAIR_RUN"); fi
# Only the initial shared-reference pass is serialised; A/B train concurrently.
(
  flock 8
  python -u run_full_h0_joint.py prepare "${args[@]}"
) 8>"$PAIR_DIR/.reference.lock"
python -u run_full_h0_joint.py "$MODE" "${args[@]}" 2>&1 | tee -a "$PAIR_DIR/$RUN_SUBDIR/console.log"
if [[ "$MODE" == all ]]; then
  (
    flock 8
    python -u compare_full_h0_joint.py --pair_dir "$PAIR_DIR" --b_run_name "$B_RUN_NAME"
  ) 8>"$PAIR_DIR/.compare.lock"
fi
