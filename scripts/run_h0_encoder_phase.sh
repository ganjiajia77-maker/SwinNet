#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
MODE=${1:-}
case "$MODE" in smoke|train|resume) ;; *) echo 'Usage: run_h0_encoder_phase.sh smoke|train|resume' >&2; exit 2 ;; esac
: "${PHASE_RUN_NAME:?Set an independent PHASE_RUN_NAME}"
DATA=${DATA:-/home/gjj/Swin-Unet-main/Massachusetts/tiff}
RESULTS=${RESULTS:-/home/gjj/SwinNet-roadbias/model_out}
PRETRAIN=${PRETRAIN:-/home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth}
H0_REFERENCE=${H0_REFERENCE:-/home/gjj/SwinNet-h0-mass/model_out/mass_h0_599a410_random512_fp32_80e_bs2_20261011_113054/best.pth}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OPENCV_LOG_LEVEL=ERROR
test -s "$H0_REFERENCE"
test -s "$PRETRAIN"
test -d "$DATA/train"
mkdir -p "$RESULTS"
run="$PHASE_RUN_NAME"
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
  --encoder_h_postmerge_phase --h0_config_checkpoint "$H0_REFERENCE" \
  --h_phase_initial_strength 0.05 --h_phase_audit_interval 50 \
  "${extra[@]}"
