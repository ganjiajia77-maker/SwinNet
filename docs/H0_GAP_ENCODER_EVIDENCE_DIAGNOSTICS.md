# H0 gap candidate and encoder evidence diagnostics

This diagnostic is based on the `codex/h0-local-gap-mass1500` implementation
and reuses its candidate GT labeler. It does not modify either checkpoint or
run surface writeback.

## What it compares

Validation 1 compares a P2-only baseline with P2 endpoints augmented by
endpoints extracted from frozen E2, H_E2, and H_E3 road probes. P2 endpoints
remain first-priority anchors. Probe maps are resized to P2 resolution and
used to add endpoint proposals and rank the line evidence; probabilities are
not summed. Every result is capped at the same candidate budget (default 64).
GT labels are applied only after candidates are generated, using the existing
`losses.gap_losses.label_edges` and `proposal_gap_coverage` functions.

Validation 2 forms one coordinate-deduplicated union candidate pool per tile.
It trains three identical 14,977-parameter MLP scorers on frozen path-pooled
features: R0 uses F2, R1 uses F2+H_E2, and R2 uses F2+H_E2+H_E3. All scorers
see the same candidates, labels, train crops, and validation tiles; excluded
feature blocks are zeroed. The head predicts candidate validity only; it does
not write corrections back to the surface.

The 39e1398 checkpoint does not contain H_E2/H_E3. Therefore P2/F2 come from
the 39e1398 Gap checkpoint, while E2/H_E2/H_E3 maps come from the separate
1af9580 encoder-exchange checkpoint and its matching frozen probe heads. This
is an external-evidence diagnostic, not evidence that the 39e1398 model has
an internal encoder-to-decoder path. The output records these checkpoint
sources explicitly.

## Server command

```bash
cd /home/gjj/SwinNet-h0-gap-encoder-probe-diagnostics
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet

DATA=/home/gjj/Swin-Unet-main/Massachusetts/tiff
GAP_CKPT=/home/gjj/SwinNet-h0-local-gap/model_out/<39e1398_run>/best.pth
PROBE_REPO=/home/gjj/SwinNet-h0-probe7-9432ac1
PROBE_CKPT=/home/gjj/SwinNet-road-skeleton/model_out/<1af9580_run>/best.pth
PROBE_HEADS=/home/gjj/SwinNet-h0-probe7-9432ac1/model_out/<probe_run>/probe_best.pth
OUT=/home/gjj/SwinNet-h0-local-gap/diagnostics/encoder_evidence_$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=2 OPENCV_LOG_LEVEL=ERROR \
python -u diagnose_gap_encoder_evidence.py \
  --root_path "$DATA" \
  --gap_model_path "$GAP_CKPT" \
  --probe_repo "$PROBE_REPO" \
  --probe_model_path "$PROBE_CKPT" \
  --probe_head_path "$PROBE_HEADS" \
  --output_dir "$OUT" \
  --device cuda:0 \
  --candidate_pool 256 \
  --candidate_budget 64 \
  --evidence_threshold 0.45 \
  --max_train_images 256 \
  --head_epochs 15 \
  --seed 1234
```

The validation pass uses the complete validation split with 512 crops at
stride 256, matching the existing 350-tile candidate audit when that split
contains 14 images. The head-training pass takes one deterministic random
512 crop from each of up to 256 training images. No test images are read.

Outputs:

- `validation1_per_tile.csv`: per-source candidate count, GT-positive and
  negative edges, verified gap coverage, and gap recall at the fixed budget.
- `validation2_head_training.json`: R0/R1/R2 validation edge AP and gap recall
  after ranking the same candidate pool, plus training loss history.
- `summary.json`: checkpoint sources and frozen candidate/label protocol.

`gap_recall` is computed over overlapping validation tiles, so the same native
gap may appear in multiple tiles. Compare every variant on these exact same
tiles; do not interpret the tile count as a unique native-gap count.
`no_local_path_negatives` is a useful proxy for unsupported straight joins,
but it is not a dedicated classifier for parallel-road confusions.
