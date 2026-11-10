#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

COMMAND=${1:-all}
VARIANT=${DIAG_VARIANT:-encoder_h}
case "$VARIANT" in
  pixel)
    RUN_POINTER=/home/gjj/h0_hpixel_latest.txt
    EVAL_POINTER=/home/gjj/h0_hpixel_eval_latest.txt
    ;;
  no_a1a2)
    RUN_POINTER=/home/gjj/h0_hpixel_noa_latest.txt
    EVAL_POINTER=/home/gjj/h0_hpixel_noa_eval_latest.txt
    ;;
  encoder_h)
    RUN_POINTER=/home/gjj/h0_encoder_h_latest.txt
    EVAL_POINTER=/home/gjj/h0_encoder_h_eval_latest.txt
    ;;
  *) echo "Unknown DIAG_VARIANT=$VARIANT" >&2; exit 2 ;;
esac

POINTER="/home/gjj/h0_hpixel_three_mode_${VARIANT}_latest.txt"
if [[ "$COMMAND" == show ]]; then
  DIAG_OUT=${DIAG_OUT:-$(cat "$POINTER")}
  exec python -u scripts/show_hpixel_three_modes.py --output_dir "$DIAG_OUT"
fi
if [[ "$COMMAND" != all && "$COMMAND" != infer && "$COMMAND" != topology ]]; then
  echo 'Expected all, infer, topology or show' >&2
  exit 2
fi

if [[ -z ${CHECKPOINT:-} ]]; then
  test -s "$RUN_POINTER"
  CHECKPOINT="$(cat "$RUN_POINTER")/best.pth"
fi
test -s "$CHECKPOINT"
if [[ "$COMMAND" == topology ]]; then
  DIAG_OUT=${DIAG_OUT:-$(cat "$POINTER")}
else
  DIAG_OUT=${DIAG_OUT:-"$(dirname "$CHECKPOINT")/three_mode_val_$(date +%Y%m%d_%H%M%S)"}
fi
FIXED_GAPS=${FIXED_GAPS:-/home/gjj/SwinNet-roadbias/model_out/full_h0_joint_pair_v1/reference/fixed_gaps_and_anchors.json}
test -s "$FIXED_GAPS"

THRESHOLD_ARGS=()
if [[ -n ${DIAG_THRESHOLD:-} ]]; then
  THRESHOLD_ARGS=(--threshold "$DIAG_THRESHOLD")
else
  if [[ -z ${VAL_SELECTION:-} ]]; then
    VAL_SELECTION=$(python - "$CHECKPOINT" "$EVAL_POINTER" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
checkpoint, pointer = map(Path, sys.argv[1:])
sha = hashlib.sha256()
with checkpoint.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024*1024), b''):
        sha.update(chunk)
paths = []
if pointer.is_file():
    value = pointer.read_text().strip()
    if value:
        paths.append(Path(value)/'val_selection.json')
paths += sorted(checkpoint.parent.glob('eval*/val_selection.json'),
                key=lambda p: p.stat().st_mtime, reverse=True)
for path in dict.fromkeys(paths):
    if not path.is_file():
        continue
    value = json.loads(path.read_text())
    if value.get('selected_on') == 'val' and value.get('checkpoint_sha256') == sha.hexdigest():
        print(path.resolve())
        break
else:
    raise SystemExit('No matching val threshold evidence. Set VAL_SELECTION or an explicit DIAG_THRESHOLD; no threshold is guessed.')
PY
    )
  fi
  test -s "$VAL_SELECTION"
  THRESHOLD_ARGS=(--threshold_selection "$VAL_SELECTION")
fi

printf '%s\n' "$DIAG_OUT" > "$POINTER"
printf '%s\n' "$DIAG_OUT" > /home/gjj/h0_hpixel_three_mode_latest.txt
echo "Diagnostic variant: $VARIANT"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $DIAG_OUT"
echo 'One frozen checkpoint; 14 validation images; one threshold; no training.'
exec python -u scripts/diagnose_hpixel_three_modes.py \
  --checkpoint "$CHECKPOINT" \
  --root_path "${DATA_ROOT:-/home/gjj/Swin-Unet-main/Massachusetts/tiff}" \
  --fixed_gaps "$FIXED_GAPS" \
  --output_dir "$DIAG_OUT" \
  --device "${DEVICE:-cuda}" \
  --command "$COMMAND" \
  --workers "${TOPO_WORKERS:-4}" \
  --topo_tool "${TOPO_TOOL:-/home/gjj/unified-road-six-4b7d37a/evaluation/road_comparison/compare_predictions.py}" \
  "${THRESHOLD_ARGS[@]}"
