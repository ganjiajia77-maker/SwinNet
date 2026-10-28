#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export PYTHONUNBUFFERED=1
DATA=${DATA:-/home/gjj/Swin-Unet-main/Massachusetts/tiff}
RESULTS=${RESULTS:-/home/gjj/SwinNet-roadbias/model_out}
V9_RUN=${V9_RUN:-$RESULTS/h0_v9_direct_20261028_135407}
MODE=${1:-all}
mkdir -p "$RESULTS"
if [[ -z ${PATH_RUN:-} ]]; then
  if [[ -s "$RESULTS/latest_path_connection_run.txt" ]]; then
    PATH_RUN=$(cat "$RESULTS/latest_path_connection_run.txt")
  else
    PATH_RUN="$RESULTS/h0_v9_path_connection_$(date +%Y%m%d_%H%M%S)"
  fi
fi
mkdir -p "$PATH_RUN"
printf '%s\n' "$PATH_RUN" > "$RESULTS/latest_path_connection_run.txt"
if [[ "$MODE" == results ]]; then
  if [[ -s "$PATH_RUN/REPORT.md" ]]; then cat "$PATH_RUN/REPORT.md"; fi
  if [[ -s "$PATH_RUN/comparison_val/REPORT.md" ]]; then cat "$PATH_RUN/comparison_val/REPORT.md"; fi
  exit 0
fi
df -h "$RESULTS"
if [[ "$MODE" == all ]]; then
  python -m unittest checks.test_path_connection 2>&1 | tee -a "$PATH_RUN/unit_tests.log"
fi

# Locate the successful fixed reference by content; never invent a new 206-gap set.
REFERENCE_SET=$(python - "$V9_RUN" "${REFERENCE_SET:-}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); explicit=sys.argv[2]
if explicit:
    path=Path(explicit)
    if not path.is_file(): raise SystemExit(f'Missing REFERENCE_SET: {path}')
    print(path.resolve()); raise SystemExit
candidates={}
for path in root.rglob('reference_set.json'):
    try:
        data=json.loads(path.read_text())
        if data.get('eligible_reference_gaps')==206 and data.get('images')==14 and data.get('p0_reference_threshold')==.45:
            candidates.setdefault(data['sha256'],[]).append(path)
    except (ValueError,KeyError,OSError): pass
if len(candidates)!=1:
    raise SystemExit('Need one unambiguous successful V9.1/V9.2 206-gap reference set. '
                     f'Found {len(candidates)} fingerprints under {root}; set REFERENCE_SET explicitly if moved.')
print(sorted(next(iter(candidates.values())),key=str)[0].resolve())
PY
)

common=(--root_path "$DATA" --output_dir "$PATH_RUN" --v9_run "$V9_RUN"
  --reference_set "$REFERENCE_SET" --device cuda --batch_size 2
  --num_workers "${WORKERS:-4}" --epochs "${EPOCHS:-20}" --val_interval "${VAL_INTERVAL:-5}"
  --seed "${SEED:-1234}" --max_endpoints "${MAX_ENDPOINTS:-32}" --max_pairs "${MAX_PAIRS:-48}"
  --max_paths "${MAX_PATHS:-128}")
if [[ ${FINAL_TEST:-1} == 1 ]]; then common+=(--final_test); fi
echo "PATH_RUN=$PATH_RUN"
echo "V9_RUN=$V9_RUN"
echo "REFERENCE_SET=$REFERENCE_SET"
case "$MODE" in
  all|smoke|candidate_audit)
    python -u run_path_connection_mass.py "$MODE" "${common[@]}" 2>&1 | tee -a "$PATH_RUN/console.log"
    ;;
  results)
    [[ -s "$PATH_RUN/REPORT.md" ]] && cat "$PATH_RUN/REPORT.md"
    [[ -s "$PATH_RUN/comparison_val/REPORT.md" ]] && cat "$PATH_RUN/comparison_val/REPORT.md"
    ;;
  *) echo 'Usage: bash scripts/run_path_connection_mass.sh all|smoke|candidate_audit|results' >&2; exit 2 ;;
esac
