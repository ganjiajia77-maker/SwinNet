#!/usr/bin/env bash
# Evaluate one trained Mass no-C checkpoint with the pinned unified v7 tool.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${1:-}
CKPT=${CKPT:?Set CKPT to the trained no-C best.pth}
DATA=${DATA:-/home/gjj/Swin-Unet-main/Massachusetts/tiff}
REPORT=${REPORT:-$(dirname "$CKPT")/eval_no_c_mass_v7}
METRICS=${METRICS:-/home/gjj/unified-road-baselines-a85dd4f}
GPU=${GPU:-2}
WORKERS=${WORKERS:-4}
FLOOR=0.10
TOOL="$METRICS/evaluation/road_comparison/compare_predictions.py"
MANIFEST="$REPORT/comparison.json"
SELECTION="$REPORT/val/val_selection.json"

case "$MODE" in
  sweep|test|topology) ;;
  *) echo "Usage: CKPT=/absolute/best.pth bash $0 sweep|test|topology" >&2; exit 2 ;;
esac
test -s "$CKPT"
test -f "$TOOL" || { echo "Missing the unified v7 evaluator: $TOOL" >&2; exit 1; }
mkdir -p "$REPORT"
cd "$REPO"
echo "Mode=$MODE; checkpoint=$CKPT; report=$REPORT"

if [[ "$MODE" != topology ]]; then
  ema=$(python - "$CKPT" <<'PY'
import sys
import torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
cfg = state["args"]
if not cfg.get("disable_stage_connectivity") or cfg.get("source_patch_size") != 1500:
    raise SystemExit("Expected the trained native-1500 H0 no-C checkpoint")
if type(cfg.get("use_ema")) is not bool:
    raise SystemExit("Checkpoint does not record a boolean use_ema")
print(int(cfg["use_ema"]))
PY
)
  WEIGHTS=()
  if [[ "$ema" == 1 ]]; then WEIGHTS+=(--require_ema); fi
fi

prepare_manifest() {
  python "$REPO/evaluation/road_comparison/build_h0_manifest.py" \
    --checkpoint "$CKPT" --data_root "$DATA" --predictions_root "$REPORT/predictions" \
    --overlap_weight_floor "$FLOOR" --output "$MANIFEST" "$@"
  python - "$MANIFEST" "$CKPT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest, checkpoint = map(Path, sys.argv[1:])
data = json.loads(manifest.read_text(encoding="utf-8"))
sha = hashlib.sha256()
with checkpoint.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        sha.update(block)
data["models"][0]["provenance"]["checkpoint_sha256"] = sha.hexdigest()
data["protocol"] = dict(
    image_size=1500, short_area_threshold=20,
    topo_seed_density_deg=0.002, topo_step_deg=0.0002,
    apls_max_control_nodes=150, apls_snap_radius=5.0,
    apls_control_sampling_seed=1234, apls_min_path_px=10.0,
    tlts_n_paths=200, tlts_seed=1234,
)
tmp = manifest.with_suffix(".tmp")
tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
tmp.replace(manifest)
PY
}

infer() {
  local split=$1 threshold=$2
  shift 2
  CUDA_VISIBLE_DEVICES="$GPU" OPENCV_LOG_LEVEL=ERROR \
  python -u "$REPO/test_image.py" \
    --root_path "$DATA" --model_path "$CKPT" \
    --cfg "$REPO/configs/swin_tiny_patch4_window7_224_lite.yaml" \
    --split "$split" --img_size 512 --source_patch_size 1500 \
    --overlap_infer --overlap_weight_floor "$FLOOR" "${WEIGHTS[@]}" \
    --output_dir "$REPORT/predictions/$split" --threshold "$threshold" --export_thresholds "$@" \
    2>&1 | tee "$REPORT/${split}_inference.log"
}

if [[ "$MODE" == sweep ]]; then
  TEST_GRID=$(python -c 'print(" ".join(f"{i/100:.2f}" for i in range(10, 66)))')
  read -ra THRESHOLDS <<< "$TEST_GRID"
  infer val 0.20 "${THRESHOLDS[@]}"
  prepare_manifest
  python -u "$TOOL" --manifest "$MANIFEST" --split val --select_threshold_only \
    --output_dir "$REPORT/val" --progress_every 1 2>&1 | tee "$REPORT/val_sweep.log"
  cat "$SELECTION"
elif [[ "$MODE" == test ]]; then
  THRESHOLD=$(python - "$SELECTION" "$MANIFEST" "$CKPT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

selection, manifest, checkpoint = map(Path, sys.argv[1:])
record = json.loads(selection.read_text(encoding="utf-8"))
model, = json.loads(manifest.read_text(encoding="utf-8"))["models"]
provenance = model["provenance"]
identity = hashlib.sha256(json.dumps(dict(name=model["name"], provenance=provenance),
                                    sort_keys=True).encode()).hexdigest()
sha = hashlib.sha256()
with checkpoint.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        sha.update(block)
chosen = record["models"][model["name"]]
if (record.get("selected_on") != "val" or chosen["identity"] != identity
        or provenance["checkpoint_sha256"] != sha.hexdigest()):
    raise SystemExit("Checkpoint or manifest does not match the validation selection")
print(chosen["threshold"])
PY
)
  echo "Validation-selected threshold: $THRESHOLD"
  infer test "$THRESHOLD" "$THRESHOLD"
  prepare_manifest --selection "$SELECTION"
  tail -n 8 "$REPORT/test_inference.log"
else
  python -u "$TOOL" --manifest "$MANIFEST" --split test --selection "$SELECTION" \
    --output_dir "$REPORT/topology_v7" --workers "$WORKERS" --progress_every 1 --resume \
    2>&1 | tee "$REPORT/topology_v7.log"
  python - "$REPORT/topology_v7/test_comparison.csv" <<'PY'
import csv
import sys

keys = ("iou", "f1", "precision", "recall", "cldice", "topo_precision", "topo_recall",
        "topo_f1", "apls_bidirectional", "tlts_infeasible")
with open(sys.argv[1], newline="", encoding="utf-8-sig") as stream:
    for row in csv.DictReader(stream):
        print(row["name"])
        for key in keys:
            print(f"{key:20s} {float(row[key]):.6f}")
PY
fi
