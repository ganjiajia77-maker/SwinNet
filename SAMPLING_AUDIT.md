# Frozen original B: final x4 and A1/A2 audit

Reuses the existing stage diagnostic's readouts, train/holdout image split,
fixed crops, normalization and checkpoint SHA. Only two readout heads continue
fitting for2 epochs (default256 crops). B is frozen/eval and hashed before/after.
The original shared post-x4 head, before-x4 head, continued shared head and
16-phase head are evaluated on the same14 native val images at512/256 stitching.
The phase head starts with exactly the original shared head's predictions.
It has16 times the parameters; this is an expressiveness diagnostic, not a fair
new architecture comparison. New Adam optimizer/LR .003 are recorded.

`x4_matrix.json` checks rank/conditioning of the actual C->16C expansion.
Pixel rearrangement is invertible. Full column rank implies the *pre-LayerNorm*
expansion/rearrangement preserves input vectors, regardless of poor linear
readout. `x4_reconstruction.json` numerically verifies this on actual features.
Its LN inverse uses saved mean/std as side information, explicitly not assumed
available to ordinary heads. These checks do not prove normalized output is
equally easy to decode, nor do they promise final road improvement.

`continued_probe_losses.csv` and `holdout_losses.json` distinguish continued
fitting from training-only overfit. Thresholds are fixed on the same training
image holdout, never selected on val. Threshold sweep .01..99 is recorded.
`comparison.csv` includes fixed-threshold and holdout-threshold readouts.

A1/A2 in this source directly modulate the *first two* residual PatchMerging
operations and supply Stage3 attention bias. The third merge receives no direct
A1/A2 map. The frozen checkpoint is evaluated normally and with merge12 road
scales zero, Stage3 bias scales zero, or both zero. It does not zero the A maps
themselves or replace them with GT. All parameters are restored even on errors.
The final B probabilities, TP/FP changes, frozen198 gap local closure changes
and bridge proxies are compared at the *same* fixed threshold. This measures
checkpoint dependence; it cannot replace a matched training-time ablation.

Default work:256 probe-training forwards +32 heldout +14*25*4 val forwards,
about1700 frozen-B tile forwards. No new B training, no feature cache, no test
threshold selection, no full TOPO/APLS calculation. `--skip_priors` reduces the
val work to one B forward per tile for an x4-only audit.

Read comparison alongside rank/reconstruction, holdout losses and parameter
scales. Shared refit improvement supports insufficient initial probe fitting;
phase-specific improvement supports limitations of shared phase-independent
readout, with a capacity caveat. If neither helps, normalized feature content,
context needs, fitting budget or final fusion can still be limiting: no automatic
conclusion of irreversible information loss is warranted.
