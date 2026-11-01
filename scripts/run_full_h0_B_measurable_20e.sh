#!/usr/bin/env bash
# Reuse the completed 20-epoch A; start/resume only the separately named B.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
export PAIR_DIR=${PAIR_DIR:-/home/gjj/SwinNet-roadbias/model_out/full_h0_joint_pair_20e_20261101_085322}
export GPU=${GPU:-1}
export EPOCHS=20
export B_RUN_NAME=B_transport_0p01_proposal_0p001
export TRANSPORT_RATE_INIT=0.01
export PROPOSAL_RATE_INIT=0.001
export OPENCV_LOG_LEVEL=${OPENCV_LOG_LEVEL:-ERROR}

python - "$PAIR_DIR" <<'PY'
import json
from pathlib import Path
import shutil
import sys

pair=Path(sys.argv[1])
config=json.loads((pair/'A/experiment.json').read_text())
complete=json.loads((pair/'A/training_complete.json').read_text())
if complete['epochs']!=20 or config['epochs']!=20:
    raise SystemExit('Expected the completed 20-epoch A; no A training will be launched.')
if not (pair/'A/eval/val_selection.json').is_file():
    raise SystemExit('A validation evaluation is incomplete.')
available=shutil.disk_usage(pair).free/2**30
if available<5:
    raise SystemExit(f'Only {available:.2f} GiB free; free at least 5 GiB before starting B.')
print(f'[CONTROL] reuse {pair}/A; 20 epochs; free disk {available:.2f} GiB',flush=True)
PY

while :; do
  FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
  PIDS=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits)
  if [ -z "$PIDS" ] && [ "$FREE" -ge 14000 ]; then break; fi
  echo "[WAIT GPU$GPU] free=${FREE} MiB; compute PIDs: $PIDS"
  sleep 30
done
exec bash scripts/run_full_h0_joint.sh B all
