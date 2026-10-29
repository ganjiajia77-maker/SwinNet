#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export PYTHONUNBUFFERED=1
DATA=${DATA:-/home/gjj/Swin-Unet-main/Massachusetts/tiff}
RESULTS=${RESULTS:-/home/gjj/SwinNet-roadbias/model_out}
RUN_DIR=${RUN_DIR:-$RESULTS/h0_v9_direct_20261028_135407}
MODE=${1:-all}
mkdir -p "$RESULTS"
if [[ -z ${GATE_BENEFIT_RUN:-} ]]; then
  if [[ -s "$RESULTS/latest_v9_gate_benefit_run.txt" ]]; then GATE_BENEFIT_RUN=$(cat "$RESULTS/latest_v9_gate_benefit_run.txt")
  else GATE_BENEFIT_RUN="$RESULTS/h0_v9_gate_benefit_$(date +%Y%m%d_%H%M%S)"; fi
fi
mkdir -p "$GATE_BENEFIT_RUN"
printf '%s\n' "$GATE_BENEFIT_RUN" > "$RESULTS/latest_v9_gate_benefit_run.txt"
if [[ "$MODE" == results ]]; then
  for file in comparison_3step/REPORT.md comparison_3step/transport_selection_screen.json steps_inference_diagnostic/status.json steps_inference_diagnostic/REPORT.md comparison_with_trained6/REPORT.md; do
    if [[ -s "$GATE_BENEFIT_RUN/$file" ]]; then cat "$GATE_BENEFIT_RUN/$file"; fi
  done
  echo "GATE_BENEFIT_RUN=$GATE_BENEFIT_RUN"
  exit 0
fi
exec 9>"$GATE_BENEFIT_RUN/.run.lock"
if ! flock -n 9; then echo "Another job is already using $GATE_BENEFIT_RUN" >&2; exit 1; fi
[[ -s "$RUN_DIR/stage_D/best.pth" && -s "$RUN_DIR/direct_val_selection.json" ]] || {
  echo "Missing original V9 checkpoint or val selection: $RUN_DIR" >&2; exit 1; }
reference=()
if [[ "$MODE" == all || "$MODE" == compare || "$MODE" == steps ]]; then
  REFERENCE_SET=$(python - "$RUN_DIR" "${REFERENCE_SET:-}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); explicit=sys.argv[2]
if explicit:
    path=Path(explicit)
    if not path.is_file(): raise SystemExit(f'Missing fixed reference: {path}')
    print(path.resolve()); raise SystemExit
found={}
for path in root.rglob('reference_set.json'):
    try:
        value=json.loads(path.read_text())
        if value.get('eligible_reference_gaps')==206 and value.get('images')==14 and value.get('p0_reference_threshold')==.45:
            found.setdefault(value['sha256'],[]).append(path)
    except (ValueError,KeyError,OSError): pass
if len(found)!=1: raise SystemExit(f'Need one unambiguous fixed206-gap fingerprint under {root}; found {len(found)}. Set REFERENCE_SET if moved.')
print(sorted(next(iter(found.values())),key=str)[0].resolve())
PY
  )
  reference=(--reference_set "$REFERENCE_SET")
fi
common=(--run_dir "$RUN_DIR" --root_path "$DATA" --output_dir "$GATE_BENEFIT_RUN" --device cuda
  --epochs "${EPOCHS:-20}" --batch_size 2 --num_workers "${WORKERS:-4}" --seed "${SEED:-1234}"
  --val_interval "${VAL_INTERVAL:-5}" --lambda_gate "${LAMBDA_GATE:-0.1}")
if [[ ${AUTO_TRAIN6:-1} == 1 ]]; then common+=(--auto_train6); fi
echo "GATE_BENEFIT_RUN=$GATE_BENEFIT_RUN"
echo "RUN_DIR=$RUN_DIR; GPU=${GPU:-1}; paired epochs=${EPOCHS:-20}; lambda_G=${LAMBDA_GATE:-0.1}"
df -h "$RESULTS"
case "$MODE" in
  all)
    python -m unittest checks.test_v9_gate_benefit 2>&1 | tee -a "$GATE_BENEFIT_RUN/unit_tests.log"
    python -u run_v9_gate_benefit.py all "${common[@]}" "${reference[@]}" 2>&1 | tee -a "$GATE_BENEFIT_RUN/console.log"
    ;;
  smoke|control|gate|compare|steps)
    python -u run_v9_gate_benefit.py "$MODE" "${common[@]}" "${reference[@]}" 2>&1 | tee -a "$GATE_BENEFIT_RUN/console.log"
    ;;
  *) echo 'Usage: run_v9_gate_benefit.sh all|smoke|control|gate|compare|steps|results' >&2; exit 2 ;;
esac
