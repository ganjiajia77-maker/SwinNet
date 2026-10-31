#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
DATA=${DATA:-/home/gjj/Swin-Unet-main/Massachusetts/tiff}
SELECTOR_RUN=${SELECTOR_RUN:-/home/gjj/SwinNet-roadbias/model_out/h0_v9_source_selector_20261030_110206}
if [[ -z ${TRANSPORT_RUN:-} ]]; then
  [[ -s "$SELECTOR_RUN/latest_actual_transport_run.txt" ]] || { echo 'Missing current S run marker; set TRANSPORT_RUN.' >&2; exit 1; }
  TRANSPORT_RUN=$(cat "$SELECTOR_RUN/latest_actual_transport_run.txt")
fi
MANIFEST="$TRANSPORT_RUN/fixed_gaps_and_anchors.json"
[[ -s "$MANIFEST" ]] || { echo "Missing fixed manifest: $MANIFEST" >&2; exit 1; }
if [[ -z ${CHOICE_RUN:-} && -s "$TRANSPORT_RUN/latest_choice_loss_run.txt" ]]; then CHOICE_RUN=$(cat "$TRANSPORT_RUN/latest_choice_loss_run.txt"); fi
REVIEW_OUT=${REVIEW_OUT:-${CHOICE_RUN:-$TRANSPORT_RUN}/fixed_gap_scene_review_v1}
mkdir -p "$REVIEW_OUT"
exec 9>"$REVIEW_OUT/.review.lock"
flock -n 9 || { echo "Already running: $REVIEW_OUT" >&2; exit 1; }
args=(--root_path "$DATA" --manifest "$MANIFEST" --transport_run "$TRANSPORT_RUN" --output_dir "$REVIEW_OUT" --context "${CONTEXT:-128}")
if [[ -n ${CHOICE_RUN:-} && -s "$CHOICE_RUN/comparison.json" ]]; then
  args+=(--choice_run "$CHOICE_RUN")
else
  echo 'Completed choice comparison not found; exporting original images/GT and available P0/S caches. No inference will be launched.'
fi
if [[ -n ${THRESHOLD:-} ]]; then args+=(--threshold "$THRESHOLD"); fi
python -u diagnostics/review_fixed_gaps.py "${args[@]}" 2>&1 | tee "$REVIEW_OUT/generation.log"
printf '%s\n' "$REVIEW_OUT" > "$TRANSPORT_RUN/latest_fixed_gap_review.txt"
tar -czf "$REVIEW_OUT.tar.gz" -C "$REVIEW_OUT" .
echo "REVIEW_HTML=$REVIEW_OUT/index.html"
echo "NATIVE_MARKERS=$REVIEW_OUT/overviews"
echo "SCENE_TEMPLATE=$REVIEW_OUT/review_template.csv"
echo "DOWNLOAD_ARCHIVE=$REVIEW_OUT.tar.gz"
