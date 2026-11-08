#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${LOCAL_CS_RUN_DIR:?Specify the completed fresh80 run directory}"
: "${SELECTION_JSON:?Specify its existing val_selection.json}"
: "${DIAG_OUT:?Specify a new independent diagnostic directory}"
REFERENCE_B=${REFERENCE_B:-/home/gjj/SwinNet-roadbias/model_out/h0_native_relation_80e_s1234_B/best.pth}
DATA=${DATA:-/home/gjj/Swin-Unet-main/Massachusetts/tiff}
FIXED_GAPS=${FIXED_GAPS:-/home/gjj/SwinNet-roadbias/model_out/full_h0_joint_pair_v1/reference/fixed_gaps_and_anchors.json}
test -s "$REFERENCE_B"
test -s "$LOCAL_CS_RUN_DIR/best.pth"
test -s "$SELECTION_JSON"
test -s "$FIXED_GAPS"
test ! -e "$DIAG_OUT" || { echo 'Use a new DIAG_OUT' >&2; exit 1; }
threshold=$(python - "$SELECTION_JSON" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['threshold'])
PY
)
thresholds=(0.2)
if python - "$threshold" <<'PY'
import sys
sys.exit(0 if abs(float(sys.argv[1])-.2)>1e-6 else 1)
PY
then
  thresholds+=("$threshold")
fi
python -u scripts/evaluate_local_cs.py \
  --comparison_kind independent \
  --reference "$REFERENCE_B" \
  --run "c_guided=$LOCAL_CS_RUN_DIR/best.pth" \
  --root_path "$DATA" --fixed_gaps "$FIXED_GAPS" \
  --diagnostic_thresholds "${thresholds[@]}" \
  --threshold_selection "$SELECTION_JSON" \
  --output_dir "$DIAG_OUT"
python scripts/show_local_cs_diagnostic.py --output_dir "$DIAG_OUT" > "$DIAG_OUT/TABLES.txt"
echo "[ALL DONE] $DIAG_OUT"
