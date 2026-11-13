#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
MODE=${1:-}
case "$MODE" in check|smoke|train|resume) ;; *) echo 'Usage: run_h0_encoder_phase.sh check|smoke|train|resume' >&2; exit 2 ;; esac
: "${PHASE_RUN_NAME:?Set an independent PHASE_RUN_NAME}"
: "${DATA:?Set DATA}"
RESULTS=${RESULTS:-$REPO/model_out}
: "${PRETRAIN:?Set PRETRAIN}"
: "${H0_REFERENCE:?Set H0_REFERENCE}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OPENCV_LOG_LEVEL=ERROR
test -s "$H0_REFERENCE"
test -s "$PRETRAIN"
test -d "$DATA/train"
mkdir -p "$RESULTS"
run="$PHASE_RUN_NAME"
if [[ "$MODE" == check ]]; then
  exec python -u scripts/check_h0_epoch_sampling.py --root_path "$DATA" \
    --seed 1234 --num_workers 4 --output "$RESULTS/${run}_sampling.json"
fi
extra=()
if [[ "$MODE" == smoke ]]; then
  run="${run}_smoke"
  extra+=(--h_phase_smoke_only)
fi
if [[ "$MODE" == resume ]]; then
  test -s "$RESULTS/$run/last.pth"
  extra+=(--resume "$RESULTS/$run/last.pth")
else
  test ! -e "$RESULTS/$run" || { echo "Output already exists: $RESULTS/$run" >&2; exit 1; }
fi
if [[ "$MODE" == train ]]; then
  test -s "$RESULTS/${run}_smoke/phase_smoke.json"
  test -s "$RESULTS/${run}_sampling.json"
  python - "$RESULTS/${run}_sampling.json" "$RESULTS/${run}_smoke/h0_reference_config.json" <<'PY'
import json,sys
from pathlib import Path
from scripts.check_h0_epoch_sampling import check_original_sources
report=json.load(open(sys.argv[1]))
config=json.load(open(sys.argv[2]))['resolved_args']
assert report['passed'] and report['worker_direct_equal'] and report['checked_unique_epochs']==80
assert report['sources_sha256_lf']==check_original_sources()
assert report['seed']==config['seed'] and report['num_workers']==config['num_workers']
assert report['persistent_workers'] is False and config['disable_persistent_workers'] is True
assert config['random_crop_train'] and config['random_crops_per_image']==1
assert Path(report['data_root']).resolve()==Path(config['root_path']).resolve()
PY
  python - "$RESULTS/${run}_smoke/phase_smoke.json" "$H0_REFERENCE" "$PRETRAIN" <<'PY'
import hashlib,json,sys
def sha(path):
    value=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()
report=json.load(open(sys.argv[1]))
assert report['passed'] and report['optimizer_steps']==0 and report['H_consumed_by_decoder']
assert report['reference_checkpoint_sha256']==sha(sys.argv[2])
assert report['pretrain_sha256']==sha(sys.argv[3]) and report['initial_strength']==.05
PY
  extra+=(--h_phase_smoke_manifest "$RESULTS/${run}_smoke/phase_initial_state.json")
fi
exec python -u train_image.py \
  --root_path "$DATA" --output_dir "$RESULTS" --run_name "$run" \
  --cfg "$REPO/configs/swin_tiny_patch4_window7_224_lite.yaml" \
  --pretrain_ckpt "$PRETRAIN" \
  --disable_persistent_workers \
  --encoder_h_postmerge_phase --h0_config_checkpoint "$H0_REFERENCE" \
  --h_phase_initial_strength 0.05 --h_phase_audit_interval 50 \
  "${extra[@]}"
