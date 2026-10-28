#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export PYTHONUNBUFFERED=1
DATA=${DATA:-/home/gjj/Swin-Unet-main/Massachusetts/tiff}
RESULTS=${RESULTS:-/home/gjj/SwinNet-roadbias/model_out}
MODE=${1:-preflight}
mkdir -p "$RESULTS"
if [[ -z ${V9_RUN:-} ]]; then
  if [[ -f "$RESULTS/latest_v9_run.txt" ]]; then
    V9_RUN=$(cat "$RESULTS/latest_v9_run.txt")
  else
    V9_RUN="$RESULTS/h0_v9_state_$(date +%Y%m%d_%H%M%S)"
  fi
fi
mkdir -p "$V9_RUN"
printf '%s\n' "$V9_RUN" > "$RESULTS/latest_v9_run.txt"
V7_CKPT=${V7_CKPT:-/home/gjj/SwinNet-roadbias/model_out/h0_reliable_v7_gtransport_20261025_232506/best.pth}
V8_CKPT=${V8_CKPT:-/home/gjj/SwinNet-roadbias/model_out/h0_reliable_v8_full_20261026_123229/best.pth}
BASELINE_LABEL=${BASELINE_LABEL:-original}

# A newly rebuilt baseline is recorded once and reused by all A/B/C/D stages.
if [[ -f "$V9_RUN/baseline_paths.json" ]]; then
  V7_CKPT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["v7"])' "$V9_RUN/baseline_paths.json")
  V8_CKPT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["v8"])' "$V9_RUN/baseline_paths.json")
  BASELINE_LABEL=rebuilt
fi

common=(--root_path "$DATA" --output_dir "$V9_RUN" --v7_checkpoint "$V7_CKPT"
        --v8_checkpoint "$V8_CKPT" --device cuda --relation_mode full
        --baseline_label "$BASELINE_LABEL" --batch_size 2 --num_workers "${WORKERS:-4}"
        --seed "${SEED:-1234}" --reference_threshold "${P0_REFERENCE_THRESHOLD:-0.45}")

run() { python -u run_reliable_v9_mass.py "$1" "${common[@]}" "${@:2}"; }
resolve_q3() {
  Q3_CKPT=$(python - "$RESULTS" "${Q3_CKPT:-}" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
explicit = sys.argv[2].strip()
if explicit and Path(explicit).is_file():
    print(Path(explicit).resolve()); raise SystemExit(0)
if explicit not in ('', '/best.pth'):
    raise SystemExit(f'Q3_CKPT does not exist: {explicit}')
marker = root/'rebuild_q3_r3_run.txt'
recorded = marker.read_text().strip() if marker.is_file() else ''
if recorded:
    path = Path(recorded)/'best.pth'
    if path.is_file():
        print(path.resolve()); raise SystemExit(0)
    raise SystemExit(f'Q3 run was recorded but has not saved best.pth yet: {path}')
candidates = sorted(p.resolve() for p in root.glob('h0_q3_r3_dual_*/best.pth') if p.is_file())
if len(candidates) == 1:
    print(candidates[0]); raise SystemExit(0)
if not candidates:
    raise SystemExit('No saved Q3 checkpoint found. Wait for Q3 training to save best.pth, or set Q3_CKPT to its full path.')
raise SystemExit('Multiple Q3 candidates found; set Q3_CKPT explicitly:\n' + '\n'.join(map(str, candidates)))
PY
  )
  echo "Q3_CKPT=$Q3_CKPT"
}
run_direct() {
  python -u run_reliable_v9_mass.py "$1" --initialization direct \
    --root_path "$DATA" --output_dir "$V9_RUN" --q3_checkpoint "$Q3_CKPT" \
    --device cuda --batch_size 2 --num_workers "${WORKERS:-4}" --seed "${SEED:-1234}" \
    --reference_threshold "${P0_REFERENCE_THRESHOLD:-0.45}" --direct_lr "${DIRECT_LR:-1e-4}" "${@:2}"
}
train_stage() {
  local stage=$1
  local extra=()
  if [[ "$stage" == C ]]; then extra=(--previous_checkpoint "$V9_RUN/stage_B/best.pth"); fi
  if [[ "$stage" == D ]]; then extra=(--previous_checkpoint "$V9_RUN/stage_C/best.pth"); fi
  run train --stage "$stage" --epochs "${EPOCHS:-20}" --val_interval 5 "${extra[@]}"
}
compare() {
  run compare --b_checkpoint "$V9_RUN/stage_B/best.pth" \
    --c_checkpoint "$V9_RUN/stage_C/best.pth" --d_checkpoint "$V9_RUN/stage_D/best.pth"
}
audit_all() {
  for stage in A_v8 B C D; do
    local threshold
    threshold=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["comparisons"][sys.argv[2]]["matched_v8_recall"]["threshold"])' \
      "$V9_RUN/comparison_val.json" "$stage")
    local extra=()
    if [[ "$stage" != A_v8 ]]; then extra=(--dynamic_checkpoint "$V9_RUN/stage_$stage/best.pth"); fi
    run audit --audit_stage "$stage" --audit_threshold "$threshold" "${extra[@]}"
  done
}

case "$MODE" in
  bootstrap)
    resolve_q3
    BASE_CKPT=$(python -c 'import torch,sys; print(torch.load(sys.argv[1],map_location="cpu",weights_only=False)["base_checkpoint"])' "$Q3_CKPT")
    [[ -s "$BASE_CKPT" && -s "$Q3_CKPT" ]] || { echo 'Missing the exact base referenced by Q3'; exit 1; }
    python -u run_reliable_directional_mass.py train --root_path "$DATA" \
      --output_dir "$V9_RUN/baseline_v7" --base_checkpoint "$BASE_CKPT" \
      --q3_checkpoint "$Q3_CKPT" --transport_gate_mode v7 --device cuda \
      --epochs "${BASELINE_EPOCHS:-20}" --val_interval 5 --batch_size 2 \
      --num_workers "${WORKERS:-4}" --seed "${SEED:-1234}"
    python -u run_reliable_v8_mass.py train --root_path "$DATA" \
      --output_dir "$V9_RUN/baseline_v8" --v7_checkpoint "$V9_RUN/baseline_v7/best.pth" \
      --relation_mode full --device cuda --epochs "${BASELINE_EPOCHS:-20}" \
      --val_interval 5 --batch_size 2 --num_workers "${WORKERS:-4}" --seed "${SEED:-1234}"
    python - "$V9_RUN" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
(root/'baseline_paths.json').write_text(json.dumps({
    'v7': str(root/'baseline_v7/best.pth'), 'v8': str(root/'baseline_v8/best.pth'),
    'label': 'rebuilt: v7 trained from scratch, v8 gate trained on that v7'}, indent=2))
PY
    ;;
  direct)
    resolve_q3
    run_direct smoke
    run_direct train --stage D --epochs "${EPOCHS:-20}" --val_interval 5
    ;;
  resume_direct)
    resolve_q3
    run_direct resume --stage D --epochs "${EPOCHS:-20}" --val_interval 5 \
      --dynamic_checkpoint "$V9_RUN/stage_D/last.pth"
    ;;
  val_direct)
    resolve_q3
    run_direct val --dynamic_checkpoint "$V9_RUN/stage_D/best.pth"
    ;;
  audit_direct)
    resolve_q3
    threshold=${AUDIT_THRESHOLD:-0.6}
    if [[ -f "$V9_RUN/direct_val_selection.json" ]]; then
      threshold=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["threshold"])' "$V9_RUN/direct_val_selection.json")
    fi
    run_direct audit --audit_stage D --dynamic_checkpoint "$V9_RUN/stage_D/best.pth" --audit_threshold "$threshold"
    ;;
  preflight|stage0|smoke) run "$MODE" ;;
  b) train_stage B ;;
  c) train_stage C ;;
  d) train_stage D ;;
  resume_b|resume_c|resume_d)
    stage=${MODE#resume_}; stage=${stage^^}
    run resume --stage "$stage" --dynamic_checkpoint "$V9_RUN/stage_$stage/last.pth" \
      --epochs "${EPOCHS:-20}" --val_interval 5
    ;;
  compare) compare ;;
  audit) audit_all ;;
  all)
    run preflight
    run stage0
    run smoke
    train_stage B
    train_stage C
    train_stage D
    compare
    audit_all
    ;;
  *) echo 'Usage: direct | resume_direct | val_direct | audit_direct | bootstrap | preflight | stage0 | smoke | b | c | d | resume_b/c/d | compare | audit | all'; exit 2 ;;
esac
echo "V9_RUN=$V9_RUN"
