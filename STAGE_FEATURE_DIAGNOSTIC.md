# Original native B stage readout diagnostic

This uses an explicitly supplied original B checkpoint. It does not modify
training, loss, architecture or weights, and rejects path/local-CS trained
checkpoints to keep the starting point clear. All network files are unchanged.

## Capture positions

`diagnostics/stage_feature_readout.py` installs removable hooks at:

- Patch embedding output.
- Encoder stages 1, 2 and 3 immediately before and after PatchMerging.
- Encoder stage 4 after its Transformer blocks.
- Bottleneck context fusion, bottleneck Transformer and normalization.
- Every decoder stage after skip concatenation/projection, after Transformer
  blocks before PatchExpand, and after PatchExpand. Decoder stage 0 is the
  initial upsample without skip; stage 3 has no internal upsample.
- Stage2 and Stage3 after their structure blocks; G2/G3 are decoder structural
  features, not encoder stage2/stage3.
- H0, H before/after Stage2 global context, H before/after Stage3 local relations.
- Final x4 upsample, H/G3 surface fusion, surface refinement, boundary correction
  and the actual surface prediction.

All resolutions and channels are taken from actual tensors and written to
stages.csv. Encoder `BasicLayer` combines Transformer blocks and downsampling,
so taking its output alone would confuse the source of a change. The same issue
applies to decoder `BasicLayer_up`. These hooks distinguish both operations.
They return None and clone detached features, preserving all full channels.

## Readouts and split

Each captured feature gets a separate 1x1 linear readout with train-only channel
mean/variance normalization. No channel reduction, new backbone or spatial
decoder is fitted. Logits are bilinearly upsampled to native 512 before BCE+Dice
road supervision. This is an independent diagnostic objective, not B training.
Architecture and loss definition are shared across heads, but parameter counts
vary with input channel count and are reported explicitly.

Defaults: 64 distinct training images x 2 fixed random crops, 3 probe epochs;
16 other training images x 2 crops choose thresholds. 32 fit crops estimate
normalization. Fixed seed1234; no augmentation; training crops never come from
validation/test. All heads share exactly the same crops, order and step count.
Validation is all 14 native1500 images using original weighted-logit 512/256
stitching. The base B is frozen/eval throughout; only the heads have an optimizer.
Before/after model hashes, unchanged buffers, absence of B gradients and zero
hook-vs-original output difference are checked. Readouts and full split metadata
are saved. Features themselves are not cached to disk.

The default work is approximately 800 frozen-B tile forwards, including probe
fitting and 14x25 validation tiles. It is not a new 80-epoch B training. GPU
runtime depends on hardware/load and should be read from progress logs. Use a
free GPU; this need not share the GPU running the new local-CS experiment.

## Interpret outputs

- `index.html`: all summary rows and up to24 balanced gap panels across images.
  RGB/GT/actual B and every readout are shown at the same native location, with
  a fixed 0..1 colour scale (blue low, red high) and white GT outlines. These are
  readout probabilities, never uncalibrated mean feature activations.
- `overviews/`: original native images with all frozen gap IDs marked.
- `predictions/<stage>/<case>.png`: native1500 8-bit probability visualization,
  not binary masks. Quantitative evaluation uses full float probabilities.
- `stage_metrics.csv`: same threshold, per-head train-holdout threshold, and
  thresholds selected to match actual B recall on training holdout. Matching is
  not guaranteed exact on val; actual val Recall/FPR is reported.
- `adjacent_stage_changes.csv`: explicitly named transitions (downsample,
  upsample, skip fusion, H context/relation updates, final refinement), not
  arbitrary neighboring rows from different branches.
- `per_image.csv`: all per-image values and sample counts. The missed-skeleton
  set is fixed from the same actual B output with a5x5 coverage neighborhood;
  difficult background is GT background3..64 native pixels from GT roads.
- `gap_status.csv`: original198 gap IDs/anchors, threshold .2 by default, local
  gap coverage and closure proxy. Local closure requires80% missing-pixel
  coverage and anchor connectivity within gap+anchor bounding box plus16pixels.
- `checks.json`, `split.json`, `holdout_threshold_curves.csv`, `stages.csv`,
  `probe_fit_losses.csv`, `readouts.pth`: evidence for reproducibility and probe
  capacity/convergence. No test-set threshold selection or full TOPO/APLS claim.

A decrease is evidence of weaker *linear decodability under this readout*, not
proof that information was irreversibly destroyed. Coarse-grid interpolation
limits localization; different channel counts change head capacity; short probe
fitting may not converge. Compare same positions, matched Recall, background
false positives, train/holdout curves and shallow skip/H branches together.
Strong deep-vs-shallow differences should not alone be attributed to downsampling.
