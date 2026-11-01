#!/usr/bin/env bash
# Restart only a B run that failed before its first checkpoint; preserve A.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
export PAIR_DIR=${PAIR_DIR:?Set PAIR_DIR to the existing completed A/B pair directory}
export GPU=${GPU:-1}
export OPENCV_LOG_LEVEL=${OPENCV_LOG_LEVEL:-ERROR}

EPOCHS=$(python - "$PAIR_DIR" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

pair=Path(sys.argv[1]).resolve()
contract=json.loads((pair/'pair_contract.json').read_text())
complete=json.loads((pair/'A/training_complete.json').read_text())
if complete['epochs']!=contract['epochs'] or not (pair/'A/eval/val_selection.json').is_file():
    raise SystemExit('A has not finished the agreed training and evaluation; leave it running.')
folder=pair/'B'
if folder.exists():
    config_path=folder/'experiment.json'
    config=json.loads(config_path.read_text()) if config_path.is_file() else {}
    fixed=(config.get('rate_initial_bias')==-13. and
           str(config.get('rate_initialization','')).startswith('final rates output weight=0;'))
    checkpoints=list(folder.glob('*.pth'))
    if not fixed:
        if checkpoints:
            raise SystemExit('B already contains model weights. Refusing to replace its experiment directory.')
        archived=pair/('B_before_identity_fix_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        folder.rename(archived)
        print('Archived failed pre-training B metadata: '+str(archived),file=sys.stderr)
print(contract['epochs'])
PY
)
export EPOCHS

while :; do
  FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
  PIDS=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits)
  if [ -z "$PIDS" ] && [ "$FREE" -ge 14000 ]; then break; fi
  echo "[WAIT GPU$GPU] free=${FREE} MiB; compute PIDs: $PIDS"
  sleep 30
done
exec bash scripts/run_full_h0_joint.sh B all
