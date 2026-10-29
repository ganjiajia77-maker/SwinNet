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
if [[ -z ${PATH_LOSS_RUN:-} ]]; then
  if [[ -s "$RESULTS/latest_v9_final_path_loss_run.txt" ]]; then
    PATH_LOSS_RUN=$(cat "$RESULTS/latest_v9_final_path_loss_run.txt")
  else
    PATH_LOSS_RUN="$RESULTS/h0_v9_final_path_loss_$(date +%Y%m%d_%H%M%S)"
  fi
fi
mkdir -p "$PATH_LOSS_RUN"
printf '%s\n' "$PATH_LOSS_RUN" > "$RESULTS/latest_v9_final_path_loss_run.txt"
if [[ "$MODE" == results ]]; then
  if [[ -s "$PATH_LOSS_RUN/comparison_val/REPORT.md" ]]; then cat "$PATH_LOSS_RUN/comparison_val/REPORT.md"; fi
  if [[ -s "$PATH_LOSS_RUN/epochs.jsonl" ]]; then tail -n 3 "$PATH_LOSS_RUN/epochs.jsonl"; fi
  echo "PATH_LOSS_RUN=$PATH_LOSS_RUN"
  exit 0
fi
exec 9>"$PATH_LOSS_RUN/.run.lock"
if ! flock -n 9; then
  echo "Another path-loss job is already using $PATH_LOSS_RUN; inspect its log before restarting." >&2
  exit 1
fi
[[ -s "$V9_RUN/stage_D/best.pth" && -s "$V9_RUN/direct_val_selection.json" ]] || {
  echo "Missing original V9 checkpoint/val selection in $V9_RUN" >&2; exit 1; }

reference=()
if [[ "$MODE" == all || "$MODE" == compare || "$MODE" == test ]]; then
  REFERENCE_SET=$(python - "$V9_RUN" "${REFERENCE_SET:-}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); explicit=sys.argv[2]
if explicit:
    path=Path(explicit)
    if not path.is_file(): raise SystemExit(f'Missing fixed reference: {path}')
    print(path.resolve()); raise SystemExit
candidates={}
for path in root.rglob('reference_set.json'):
    try:
        data=json.loads(path.read_text())
        if data.get('eligible_reference_gaps')==206 and data.get('images')==14 and data.get('p0_reference_threshold')==.45:
            candidates.setdefault(data['sha256'],[]).append(path)
    except (ValueError,KeyError,OSError): pass
if len(candidates)!=1:
    raise SystemExit(f'Expected one fixed 206-gap reference fingerprint in {root}; found {len(candidates)}. Set REFERENCE_SET if moved.')
print(sorted(next(iter(candidates.values())),key=str)[0].resolve())
PY
  )
  reference=(--reference_set "$REFERENCE_SET")
fi
common=(--root_path "$DATA" --output_dir "$PATH_LOSS_RUN" --v9_run "$V9_RUN" --device cuda
  --epochs "${EPOCHS:-20}" --batch_size 2 --num_workers "${WORKERS:-4}" --seed "${SEED:-1234}" --val_interval "${VAL_INTERVAL:-5}"
  --lambda_pos "${LAMBDA_POS:-0.1}" --lambda_neg "${LAMBDA_NEG:-0.1}"
  --lambda_bg_path "${LAMBDA_BG_PATH:-0.05}" --lambda_rank "${LAMBDA_RANK:-0.1}")
echo "PATH_LOSS_RUN=$PATH_LOSS_RUN"
echo "V9_RUN=$V9_RUN; GPU=${GPU:-1}"
df -h "$RESULTS"
case "$MODE" in
  all)
    python -m unittest checks.test_v9_final_path_loss 2>&1 | tee -a "$PATH_LOSS_RUN/unit_tests.log"
    python -u run_v9_final_path_loss.py all "${common[@]}" "${reference[@]}" 2>&1 | tee -a "$PATH_LOSS_RUN/console.log"
    ;;
  smoke|train|compare|test)
    python -u run_v9_final_path_loss.py "$MODE" "${common[@]}" "${reference[@]}" 2>&1 | tee -a "$PATH_LOSS_RUN/console.log"
    ;;
  *) echo 'Usage: run_v9_final_path_loss.sh all|smoke|train|compare|test|results' >&2; exit 2 ;;
esac
