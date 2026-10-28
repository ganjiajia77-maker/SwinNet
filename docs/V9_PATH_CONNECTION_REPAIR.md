# Candidate connection discrimination and controlled nonlocal repair

This branch extends the completed **direct_random_downstream** V9. It is not
a strict ablation of the original trained V8. BASE/Q3 and the V9 checkpoint
are loaded strictly and fingerprinted. The three original core source files,
including `compose_surface_logits` and `dynamic_losses`, remain unchanged.

## Computation and coordinates

For native 1500 images, training uses one random native512 crop per image per
epoch. Val/test use native512 windows with stride256 and weighted-logit fusion.
The work size is read from H (normally128 for input512), not hard coded.
Existing gate relation center32, context32, and aggregated pair32 embeddings
are reused. No new H/H3 encoder is added. The symmetric descriptor has
13*32+23=439 values; the default 64-hidden MLP has 33,263 parameters, plus one
trainable nonlocal strength. `smoke.json` records actual counts, shapes, timing
and GPU memory, rather than assuming these numbers for a different backbone.

Predicted P0 skeleton endpoints are sampled at .30/.45/.60. Pair candidates
and weaker one-sided extensions are interleaved; quotas preserve length bins
and thresholds. Near pairs use the original straight local rule. Other pairs
use bounded eight-neighbor A* with existing W, P0, Q and endpoint-conditioned
semantic support. A second path can repel the first corridor. There is no GT
input to generation or repair. Search caps and ambiguous labels limit recall.
Lengths, widths and endpoints are stored in work coordinates and native pixel
units. For native images work->crop conversion is `(coord+.5)*scale-.5`;
the stored tile/crop origin maps this to the native1500 image. Training caches
are unaugmented, so crop origins retain their native meaning.

GT labeling occurs only after predictions have generated a path. It checks
native-coordinate skeleton snaps, graph detours, path deviation, background
fraction and P0 missing regions. Labels are proxies; uncertain cases are -1
and ignored. Negative subtypes include parallel/lateral bridges, shortcuts,
false/isolated extensions and similar linear background. Endpoint ordering
does not affect the descriptor: mean, absolute difference and product are
used, and left/right environment features are symmetric.

Each path contributes `q*sqrt(Ra*Rb)*C*relu(Psrc-U3)` in a finite Gaussian
corridor. Max fusion avoids summing overlapping paths. The learned sigmoid
strength is in [0,1], multiplied by a training warmup that starts at zero.
Consequently U_final remains in [0,1]. Zero strength bypasses recomposition
and returns the original V9 logits exactly. Empty candidates also preserve V9.
Path selection is detached; q, source probabilities/features and updates keep
their gradients. E/F apply candidate-local negative confidence only to alpha;
the three dynamics steps are then recomputed, preserving original B/A weights.

## Stages, losses, automatic evidence gates

- A: exact frozen original V9 baseline.
- B: only PathRelationHead, matched connection BCE; final P1 exactly unchanged.
  Frozen candidate features are cached. Source images are partitioned inside
  the original train split with a deterministic 20% head holdout. V9 itself
  had already been trained on that original split; this is a head holdout,
  not a newly unseen backbone dataset. Selection uses **val** natural AP.
- C: frozen V9; train head and scalar strength with existing segmentation and
  continuity plus relation BCE, warming the nonlocal correction from zero.
- D: the C architecture plus positive-path continuity and GT-background hard-negative penalties.
  GT missing skeleton pixels covered by positive candidates are removed from
  the original L_cont and supervised once by L_path. Original L_seg weights
  are preserved. Default path and old-cont coefficients are both .2.
  C and D both start from the same B checkpoint, with the same20-epoch budget,
  crop/RNG sequence, optimizer and warmup. D does not receive extra C pretraining.
- E: independent frozen-weight intervention on D; no additional training.
  Only high-confidence negative paths affect alpha, with rho=.5 and q<.1
  producing a continuous confidence. Outside candidate corridors it is zero.
- F: lower-LR R/W/G/state_gate/rates fine tuning, plus head/strength. BASE/Q3,
  original visual projection and B/A weights remain frozen.

The default is20 epochs **per trained stage** (B/C/D, optionally F), batch2,
accumulation1, FP32 and no EMA. This is more than one20-epoch run.

The `all` command is noninteractive. It first runs a real-batch smoke and
caches one deterministic unaugmented512 training crop per source image; this
is **not** another full1108*25-window scan or a native full-image gap census.
Boundary-truncated gaps are excluded from full-coverage rates. All val windows
are used. Crop coverage diagnostics and classification metrics are explicit
about their sample denominators. A full stitched train census would require
an additional scan; these sampled-crop numbers must not be called that census.
The full candidate coverage gate counts verified pair paths with
sqrt(Ra*Rb)>=.05; geometry-only one-sided extensions are reported separately
and cannot inflate the two-source repair discovery rate.

Repair training is blocked automatically when fewer than8 sampled32+ gaps
are eligible or fewer than5% have >=99% geometric candidate coverage. After
B, both val and the train-image holdout need at least8 positive/8 negative
candidates, ROC-AUC>=.60, and AP above natural prevalence. These are disclosed
preconditions, not proof of efficacy. All thresholds are configurable and
recorded; a failed gate writes REPORT.md and pipeline_state.json, exits2,
and never asks for Enter. F runs only when E at matched V9 Recall closes more
gaps, adds no bridge proxies, and has no lower bidirectional approximate APLS.
Otherwise B-E results are complete and F is explicitly marked skipped.

## Evaluation and resume

V9 and all new variants use the same transparent val selection rule: maximum
global IoU over thresholds .05:.01:.95. This deliberately differs from the
older weighted topology score and is re-applied to V9 as well. Epoch selection
for C/D/F also uses val global IoU. Final comparisons include own thresholds,
the same V9 threshold, fixedP0.45, fine matched Recall and foreground area
(1e-4 grid), actual matching residuals, IoU/F1/Precision/Recall, clDice,
both directional and bidirectional approximate APLS, short components,
addedTP/FP and fixed206-gap closure with length bins and background bridges.
The P0/GT/skeleton fingerprint must equal the successful V9.1/V9.2 reference.
These APLS, gap and bridge measures remain **proxies**, not official SpaceNet
graph evaluation. The new classifier's own metrics are measured; .846 from
the old diagnostic classifier is never reused as its performance.

After val selection is frozen, `FINAL_TEST=1` (the launcher default) evaluates
only original V9 and the selected model on test using val thresholds and
val matching points. It does not rematch or select on test. Set FINAL_TEST=0
to finish with validation. Gates that block earlier stages also block test.

Caches and per-image probability files are atomic and reused on restart.
Training saves best/last plus optimizer/RNG each epoch and resumes at epoch
boundaries. A mid-epoch interruption reruns that epoch. Source hash, code,
candidate protocol and settings mismatches fail loudly. Checkpoints store
new heads and, for F, the original downstream modules; exact BASE/Q3 and V9
remain dependencies and must be preserved. Repeating the same shell command
uses `latest_path_connection_run.txt`; set PATH_RUN explicitly for a new run.

Run `bash scripts/run_path_connection_mass.sh all`. It resolves the existing
206-gap reference, logs all output, performs the checks/stages/comparisons,
and writes REPORT.md, comparison_val, classifier reports, cached path geometry,
smoke parameter/gradient/memory reports, representative images and optional
test_final. No scientific improvements are claimed before those runs finish.
