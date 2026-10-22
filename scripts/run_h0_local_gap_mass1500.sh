#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=${DATA:-/home/gjj/Swin-Unet-main/Massachusetts/tiff}
RESULTS=${RESULTS:-/home/gjj/SwinNet-roadbias/model_out}
GPU=${GPU:-0}
PRETRAIN=${PRETRAIN:-/home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth}
MODE=${1:-train}
CONFIG=${LOCAL_GAP_CONFIG:-}
if [[ -z "$CONFIG" ]]; then CONFIG='{}'; fi
mkdir -p "$RESULTS"
MARKER="$RESULTS/latest_h0_local_gap_run.txt"
if [[ "$MODE" == train ]]; then
  SEED=${SEED:-1234}
  RUN=${RUN:-h0_local_gap_mass1500_fp32_noema80_b2_acc1_s${SEED}_$(date +%Y%m%d_%H%M%S)}
else
  RUN=${RUN:-$(cat "$MARKER")}
fi
RUN_DIR="$RESULTS/$RUN"
cd "$REPO"
case "$MODE" in
  train)
    [[ ! -e "$RUN_DIR" ]] || { echo "Existing run: $RUN_DIR; use resume"; exit 1; }
    printf '%s\n' "$RUN" > "$MARKER"
    python -u mass_h0.py train --root_path "$DATA" --output_dir "$RESULTS" \
      --run_name "$RUN" --gpu "$GPU" --seed "$SEED" --pretrain_ckpt "$PRETRAIN" \
      --local_gap_config "$CONFIG" 2>&1 | tee "$RESULTS/$RUN.log"
    ;;
  resume)
    python -u mass_h0.py resume --root_path "$DATA" --output_dir "$RESULTS" \
      --model_path "$RUN_DIR/last.pth" --gpu "$GPU" 2>&1 | tee -a "$RESULTS/$RUN.log"
    ;;
  sweep|val)
    AUDIT=${AUDIT:-$RUN_DIR/unified_eval_$(date +%Y%m%d_%H%M%S)}
    printf '%s\n' "$AUDIT" > "$RUN_DIR/latest_unified_audit.txt"
    python -u mass_h0.py sweep --root_path "$DATA" --output_dir "$AUDIT" \
      --model_path "$RUN_DIR/best.pth" --gpu "$GPU" 2>&1 | tee "$RUN_DIR/sweep_$(date +%Y%m%d_%H%M%S).log"
    ;;
  test|topology)
    AUDIT=${AUDIT:-$(cat "$RUN_DIR/latest_unified_audit.txt")}
    python -u mass_h0.py "$MODE" --root_path "$DATA" --output_dir "$AUDIT" \
      --model_path "$RUN_DIR/best.pth" --gpu "$GPU"
    ;;
  diagnose)
    AUDIT=${AUDIT:-$(cat "$RUN_DIR/latest_unified_audit.txt")}
    OUT=${OUT:-$RUN_DIR/gap_writeback_diag_$(date +%Y%m%d_%H%M%S)}
    CUDA_VISIBLE_DEVICES="$GPU" python -u diagnose_local_gap.py --root_path "$DATA" \
      --model_path "$RUN_DIR/best.pth" --selection "$AUDIT/val/val_selection.json" \
      --output_dir "$OUT" --split "${SPLIT:-test}"
    ;;
  *) echo 'Usage: train | resume | sweep | test | topology | diagnose'; exit 2 ;;
esac
