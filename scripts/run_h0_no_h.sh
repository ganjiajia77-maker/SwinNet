#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${NO_H_RUN_NAME:?Set an independent NO_H_RUN_NAME}"
export PHASE_RUN_NAME="$NO_H_RUN_NAME"
export NO_H_ABLATION=1
exec bash "$REPO/scripts/run_h0_encoder_phase.sh" "${1:-}"
