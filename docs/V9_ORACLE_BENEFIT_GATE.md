# V9 detached transport-benefit supervision

This branch starts from V9/V9.2, excluding the previous four candidate path
losses and nonlocal path-repair experiment. No new learned module is added.
The source is the same `direct_random_downstream` V9 checkpoint that generated
the diagnostics, with exact BASE/Q3/code identity checks and fixed206-gap P0
validation reference. Original core forward source and parameters are reused.

## Labels, logits and matching

Each real forward uses its own predicted G and actual three-step states.
Afterward, a no-grad side calculation constructs:

`off = U + proposal_delta + anchor_delta`

`on = off + transport_rate*(M-U)`

`d = BCE(clamp(off), GT) - BCE(clamp(on), GT)`

Labels are1 if d>1e-7,0 if d<-1e-7. Absolute pre-G action below1e-5, GT area
fractions between.05 and.75, and near ties are excluded. Negative actions are
judged by the same BCE benefit, not assumed useless. The side calculation
never replaces G in the forward. States, targets, selection and matching are
detached. The live logit is reconstructed as the original gate logit plus
semantic delta plus state delta; its sigmoid must exactly equal the actual G.
No `logit(sigmoid(G))` inversion is used, so BCEWithLogits retains useful
gradients when sigmoid probabilities numerically saturate.

Matching is within image, step and action sign, then exact V and factor-of-two
absolute-action bins, with a maximum256 pairs per sign per image/step. Both
class means are equally weighted, and all K step terms are equally averaged.
An empty matched step contributes an explicit zero. There is no hidden
unmatched/single-class fallback. Counts and match-quality differences are
recorded; a gate epoch with no matched labels writes failure diagnostics and
is not marked as a successful training epoch.

`L = L_seg + .2 L_cont + .1 L_G` is the default. No path, ranking, gate-road,
R/B/A auxiliary loss is included. Lambda_G is configurable and not adjusted
automatically. Per-module norms of weighted L_G, L_seg and L_seg+.2L_cont,
and their ratios, are measured at the first matched batch each epoch.

## Paired training

Control3 and gate3 both start from the exact original V9, with fresh identical
AdamW learning-rate groups, original stage-D trainable scope, same seed,
native1500->one random512 crop/image/epoch, FP32, no EMA, physical batch2 and
accumulation1. BASE/Q3 remain frozen. Each continuation runs20 epochs by
default, so the paired experiment requires40 total training epochs. A
dedicated per-epoch DataLoader generator makes ordering and worker seeds
independent of loss computations. Complete case-order and full augmented
image-tensor hashes must agree in every paired epoch. Label matching uses an
independent local NumPy generator. Optimizer/RNG/epoch records permit restart
from the last completed epoch; duplicate launches are prevented with flock.
Benefit checkpoints use a distinct version with an explicit trained step
count. Legacy V9 loaders reject them rather than silently interpreting a
trained6-step checkpoint as3 steps. Use this branch's loader/launcher.

Checkpoint selection uses the same original V9 weighted score at val.60.
Final threshold selection uses the original weighted score over.35:.01:.75
for each continuation, matching the original V9 protocol. Saved probability
maps avoid repeating GPU inference when computing the curve. Comparisons
include own thresholds, same threshold, fine matched Recall and foreground
area, IoU/F1/Precision/Recall, clDice, directional/bidirectional approximate
APLS, fixed206-gap closure and length bins, bridge proxies, addedTP/FP and
short components. Test is not accessed by this experiment.

## Decisions and conditional step experiments

The full14-image val audit uses all windows with first-window ownership of
work-cell centers. It records natural and matched gate acceptance, per-step
hist-AUC and exact pooled matched ROC/PR-AUC,
actual transport BCE harm/gain, signed positive/negative actions, and regional
V/G/potential/actual transport/proposal/state changes/generated sources.
Matched-Recall recovered, unrecovered, missing-center, hard-background,
isolated-FP and false-bridge regions are reported. Missing-center components
are also split into3-7/8-15/16-31/32+ pixel bins; these are not the same
denominator as the206 closure-eligible gaps. Representative state sequences,
G, V, transport and GT display panels/raw arrays are saved. GT ownership/labels affect
statistics only. Different models create different states/actions, so these
are **on-policy** populations, not a fixed-input gate-only causal comparison.
The same-sign V/action matching reduces, but does not eliminate, that limitation.

The automatic val-only screening requires higher exact matched decision
ROC-AUC (so a uniform G decrease alone does not pass), lower harmful acceptance and
per-action BCE harm, no lower helpful acceptance, fewer bridge proxies and no
fewer fixed reference closures at matched Recall (residual difference<=.001)
versus the equal-budget control. This conservative rule is explicitly a
screening rule rather than a universal definition of model quality.

Only if that screen passes, the same trained gate3 weights are run for3/6/12
steps. A forward hook records actual U0..UK; rates, M, source support and
state gate are recomputed by the original shared module at each step. P1 uses
the actual last state; the legacy core `u3` key is not mistaken for U6/U12.
The inference-only report cannot establish a trained architecture benefit.

If6 inference steps close more gaps with no more bridges, no lower
bidirectional approximate APLS and no more per-action BCE harm at matched
Recall, `AUTO_TRAIN6=1` (default) starts a fresh20-epoch **gate6 from original
V9**, with the same gate loss, data/augmentation schedule and budget as gate3.
The trained models are then compared separately. There is no12-step training.
If either screen fails, the skipped stage and reason are saved; no Enter or
manual approval is requested. Set AUTO_TRAIN6=0 to stop at inference diagnostics.

Whole-GT-path L_cont replacement is deliberately deferred. Existing path
candidate limitations cannot be solved merely by turning G up. Approximate
APLS, closure and bridge measures remain proxies, not official graph metrics.

Run `bash scripts/run_v9_gate_benefit.sh all`; use `results` to view the compact
reports. The launcher remembers its output root and resumes the same run.
