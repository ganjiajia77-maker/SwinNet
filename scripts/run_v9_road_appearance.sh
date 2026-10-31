#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export PYTHONUNBUFFERED=1
export OPENCV_LOG_LEVEL=ERROR
SOURCE_RUN=${SOURCE_RUN:-/home/gjj/SwinNet-roadbias/model_out/h0_v9_source_selector_20261030_110206/actual_transport_20261030_181042}
RESULTS=${RESULTS:-/home/gjj/SwinNet-roadbias/model_out}
MODE=${1:-preview}
POINTER="$RESULTS/latest_v9_road_appearance_pair.txt"
mkdir -p "$RESULTS"
if [[ -z ${PAIR_DIR:-} ]]; then
  if [[ -s "$POINTER" ]]; then PAIR_DIR=$(cat "$POINTER"); else PAIR_DIR="$RESULTS/v9_frozen_road_appearance_$(date +%Y%m%d_%H%M%S)"; fi
fi
mkdir -p "$PAIR_DIR"
printf '%s\n' "$PAIR_DIR" > "$POINTER"
if [[ "$MODE" == results ]]; then cat "$PAIR_DIR/comparison/REPORT.md";echo "PAIR_DIR=$PAIR_DIR";exit 0; fi
exec 9>"$PAIR_DIR/.run.lock"
flock -n 9 || { echo "Already running: $PAIR_DIR" >&2;exit 1; }
common=(--run_dir "$SOURCE_RUN" --device cuda --preview_dir "$PAIR_DIR/preview24")
# Inherit the original S_selector budget from checkpoint metadata, not a stale EPOCHS variable.
if [[ -n ${APPEARANCE_EPOCHS:-} ]]; then common+=(--epochs "$APPEARANCE_EPOCHS"); fi
case "$MODE" in
 preview)
  python -u run_v9_road_appearance.py preview "${common[@]}" --output_dir "$PAIR_DIR/augmentation_on" --road_appearance_aug
  tar -czf "$PAIR_DIR/preview24.tar.gz" -C "$PAIR_DIR/preview24" .
  echo "PREVIEW_ARCHIVE=$PAIR_DIR/preview24.tar.gz"
  ;;
 pair)
  python -u run_v9_road_appearance.py preview "${common[@]}" --output_dir "$PAIR_DIR/augmentation_on" --road_appearance_aug
  tar -czf "$PAIR_DIR/preview24.tar.gz" -C "$PAIR_DIR/preview24" .
  echo "PREVIEW_ARCHIVE=$PAIR_DIR/preview24.tar.gz"
  python -u run_v9_road_appearance.py all "${common[@]}" --output_dir "$PAIR_DIR/control_off" 2>&1 | tee -a "$PAIR_DIR/control_off.log"
  python -u run_v9_road_appearance.py all "${common[@]}" --output_dir "$PAIR_DIR/augmentation_on" --road_appearance_aug 2>&1 | tee -a "$PAIR_DIR/augmentation_on.log"
  python -u compare_road_appearance.py --control_run "$PAIR_DIR/control_off" --augmented_run "$PAIR_DIR/augmentation_on" --output_dir "$PAIR_DIR/comparison"
  ;;
 on|off)
  if [[ "$MODE" == on ]]; then extra=(--road_appearance_aug);folder=augmentation_on;else extra=();folder=control_off;fi
  python -u run_v9_road_appearance.py all "${common[@]}" --output_dir "$PAIR_DIR/$folder" "${extra[@]}" 2>&1 | tee -a "$PAIR_DIR/$folder.log"
  ;;
 compare)
  python -u compare_road_appearance.py --control_run "$PAIR_DIR/control_off" --augmented_run "$PAIR_DIR/augmentation_on" --output_dir "$PAIR_DIR/comparison"
  ;;
 test)
  python -u run_v9_road_appearance.py test "${common[@]}" --output_dir "$PAIR_DIR/control_off"
  python -u run_v9_road_appearance.py test "${common[@]}" --output_dir "$PAIR_DIR/augmentation_on" --road_appearance_aug
  ;;
 *) echo 'Usage: preview|pair|on|off|compare|test|results' >&2;exit 2;;
esac
