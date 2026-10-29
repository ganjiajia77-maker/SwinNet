# V9 final-surface connection supervision

This continues the existing `direct_random_downstream` V9 from its exact
checkpoint, BASE and Q3. The forward model is unchanged; no path is inserted
into P1 at inference. Existing stage-D R/W/G/rates/B/A and visual parameters
are trainable with their original V9 learning rates; BASE/Q3 stay frozen.

## Prediction-first proposal, GT-only annotation

During training, the existing candidate generator sees P0, R, W, Q3 and the
reused H/H3 semantic center/context. It produces endpoint pairs and bounded
curved work-grid paths without GT. Only afterwards does the training mask and
skeleton label them using candidate-to-GT geometry, GT background and a
bounded GT graph route. Connected endpoints alone are insufficient: long GT
detours and background crossings can be negative shortcuts. Uncertain paths
are ignored. A positive path is supervised on the **actual GT skeleton route**
only after the candidate and GT path agree in both directions and cover a GT
missing region; no straight line is imposed on a curved positive. Negative
path samples are on the predicted candidate line, with GT-confirmed background
samples kept separate from real roads and endpoint neighborhoods.

The candidate budget/label rules are inherited from the prior V9.2 experiment;
the reported 1/799 proxy long-gap coverage still limits which positives can be
supervised. No hidden 5% gate stops this requested training run. Epoch logs
record labeled positives, negatives, uncertain cases, matched ranking pairs,
length bins and positive GT skeleton overlap. Zero candidate terms remain zero
and must not be interpreted as successful learning.

## Loss and matching

For each path `s=-tau*(logsumexp(-P1/tau)-log(n))`; the score is clamped only
for finite logarithms. Terms are `-log(s_pos)`, `-log(1-s_neg)`, mean
`-log(1-P1)` on confirmed background, and `softplus((m-s_pos+s_neg)/T)` on
same-image one-to-one candidate pairs matched by length, endpoint confidence,
source confidence, distance, candidate type and Q. Unmatched samples still
contribute their relevant positive/negative losses but not ranking.

`L_total=L_seg + 0.2*L_cont + .1*L_pos + .1*L_neg + .05*L_bg + .1*L_rank`
uses the saved original V9 segmentation/continuity factors, with each new
weight configurable. The original L_cont is intentionally preserved, so some
positive skeleton pixels can receive both L_cont and L_pos. Overlap is logged;
the effective supervision weight must be reviewed in the results rather than
assumed harmless. No additional gate-label or independent road head is added.

The path extraction/GT labeling are non-differentiable, as intended. Scores
are sampled directly from live `P1`, so gradients can reach U3 and the original
R/W/G/rates/B/A. First labeled-batch gradient norms per epoch are logged; they
may be zero if a module cannot influence a given path. A real-batch smoke probe
updates weights temporarily, restores them and records counts, finite loss,
gradient norms, duration and GPU peak memory.

## Running and evaluation

`bash scripts/run_v9_final_path_loss.sh all` tests, runs a real-batch smoke,
trains20 epochs then performs validation comparison. Physical batch2,
accumulation1, FP32, no EMA and seed1234 match V9; this is20 additional epochs
from V9 best, not a retraining of BASE/Q3. Candidate search happens online on
each random crop with current model predictions and can be CPU intensive.
Epoch checkpoints and RNG support restart from the last completed epoch.
Atomic per-image probability caches support restarting validation. The launcher
remembers the output path, so repeating the same command resumes the same run.
The launcher takes a per-run `flock` to prevent accidentally starting two jobs
on the same checkpoint directory. It does not terminate any other process.

Val checkpoint selection during training uses the original topology-weighted
V9 score at threshold .60. The final threshold scan instead applies the same
max-global-IoU val rule to P0, original V9 and the new model on .05:.01:.95.
It reports own thresholds, same threshold, fixed .45, matched Recall and area;
the P0/GT/skeleton fingerprint must equal the fixed206-gap reference. APLS is
approximate and gap closure/background bridges are proxies. Original V9 is a
fixed checkpoint comparison; an equal-budget continuation without path losses
would still be needed to isolate the effect of the new losses from extra epochs.

`bash scripts/run_v9_final_path_loss.sh test` is a separate final action after
val thresholds are frozen; it uses those val-selected work points without
matching or tuning on test. Test is not used during training/selection.
