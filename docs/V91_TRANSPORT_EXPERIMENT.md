# V9.1 fixed-weight transport and matched gate experiment

The original V9 checkpoint, BASE/Q3, dynamics and loss functions are untouched.
All V9-dependent imports still hash the unmodified V9 training source files.
The intervention, candidate construction and continuations live in new files.

`counterfactual` first tests original V9, state-gate residual zero, fixed
`SourceSupport_k=R`, and transport disabled. All use the same frozen inputs,
full 14 validation images, original selection threshold and the fixed P0
reference. Each variant is also matched to original V9 Recall and foreground
area with a 0.0001 threshold grid; residual mismatches are reported. By default
the audit requires exactly 206 eligible reference gaps. A mismatch is a protocol
failure, not a reason to silently use a new gap collection. It also decomposes
the saved threshold-selection score, including a diagnostic-only no-bridge
penalty curve. Existing V9 gap and bridge definitions are proxies and APLS is
the raster approximation, not official SpaceNet APLS.
The audit writes a fingerprint of ordered image identities, P0 binary masks,
GT masks and GT skeletons. The later gate comparison must reproduce this
fingerprint. Previous reports without such a fingerprint cannot retrospectively
prove that their 206 gaps were the same instances.

`candidates` uses **train** only. At the 128-scale work grid a positive label
requires GT road coverage >=.75, frozen P0 below its training reference, and
positive **pre-gate** transport potential `transport_rate * (M - U)` above
1e-5 with a source. Candidate selection deliberately does not use the realized
update `G * transport_rate * (M - U)`: doing so would hide useful candidates
whenever current G is low, preventing L_gate from learning to accept them.
The candidate CSV records both potential and realized updates. Negative labels
require GT coverage <=.05, the same executable positive transport potential, and either
proximity to GT road, Q>=.5, an isolated-addition proxy, or a false-bridge
proxy. Invalid mixed GT pixels are excluded. The bridge/island proxies are
estimated from P0 and the validation-selected P1 threshold in the work grid;
they are not certified false GT connections. Such uses of GT are restricted to
training-set labeling and diagnostics. There is no GT in model forward.

Matching occurs per image and step, in coarse bins for source V, positive
pre-gate transport magnitude, distance to a P0 road, and action-component area;
matching is one-to-one with randomized stratum order, and hard bridge/island
proxy examples have higher negative selection probability. The output includes
`matched_candidates_train.csv` with coordinates, pairs, labels, G, proxies and
matching attributes; `candidates_train.json` records counts before/after,
quality, no-match batches, and GT missing-skeleton component coverage for
3–7/8–15/16–31/32+ pixel-area bins. Candidate component area is only a
proxy for path length. Hitting one GT skeleton component is **not** GT
connection-pair recall, and neither parallel-road errors nor visual classes
are automatically certified. Inspect candidate scarcity and matching quality
before investing in training.

For the independent L_gate experiment, the control and gate continuation
start from the exact same V9 best checkpoint, same full training data, seed,
augmentation, batch, epoch budget, learning rates, fresh AdamW, and unchanged
validation selection rule. Control uses `L_seg+0.2*L_cont`; gate adds a
balanced BCE of matched G values, with lambda default 0.1. Labels and sample
matching are detached; gradients flow through G and the existing model.
If there are no matched candidates across an epoch, gate training fails rather
than silently doing only the base loss. Training refuses missing/partial or
inconsistent train-set candidate audits. Train-candidate AUC is a 100-bin
estimate on a balanced, matched sample and must not be presented as unbiased
full-population or val/test performance.
The matched CSV is a snapshot from the initial V9 and audits candidate quality;
during both continuations candidates are recomputed online from each model's
current predictions using the same algorithm. Later control and gate candidates
can therefore differ despite the same seed and augmented image. Resume restores
the saved optimizer and RNG states.

After both continuations, `compare` scans val thresholds independently and
compares original/control/gate at their own thresholds, the common original
threshold, matched Recall, and matched foreground area. All outputs use the
same fixed P0 gap set. Exact residuals are shown; no assertion of exact
matching is made. GT paths for path-ranking supervision are deliberately not
implemented in this round; use the requested separate experiment only if
L_gate improves local acceptance and the closure problem remains.
The comparison writes `same_position_gate_regions.csv` for G, V, M-U and
transport/proposal effects in regions fixed by the original V9 predictions.
These val region summaries are not matched path pairs or proof of a unique
error source.

## Server command order

From the extracted repository, activate the existing `swinunet` environment,
set `GPU=1`, and run each command serially:

```bash
export RUN_DIR=/home/gjj/SwinNet-roadbias/model_out/h0_v9_direct_20261028_135407
export DATA=/home/gjj/Swin-Unet-main/Massachusetts/tiff
bash scripts/run_v91_mass.sh counterfactual
bash scripts/run_v91_mass.sh candidates
bash scripts/run_v91_mass.sh train_control
bash scripts/run_v91_mass.sh train_gate
bash scripts/run_v91_mass.sh compare
```

The wrapper writes a unique timestamped output directory for each diagnostic
and updates `latest_counterfactual.txt`, `latest_candidates.txt`, and
`latest_compare.txt` only after that step succeeds. If a diagnostic step fails,
rerun it; old files are kept. A continuation failure should use
`resume_control` or `resume_gate`, with the same `RUN_DIR` and experiment root.
The latest experiment root is recorded in
`$RUN_DIR/latest_v91_root.txt`. This protocol does not remove or overwrite the
original V9 run.

To inspect results after the counterfactual and candidate audit:

```bash
V91_ROOT=$(cat "$RUN_DIR/latest_v91_root.txt")
CF=$(cat "$V91_ROOT/latest_counterfactual.txt")
CAND=$(cat "$V91_ROOT/latest_candidates.txt")
cat "$CF/REPORT.md"
cat "$CF/threshold_score_components.csv"
cat "$CAND/candidates_train.json"
```

After both continuations finish, inspect:

```bash
V91_ROOT=$(cat "$RUN_DIR/latest_v91_root.txt")
COMPARE=$(cat "$V91_ROOT/latest_compare.txt")
cat "$COMPARE/REPORT.md"
cat "$COMPARE/comparison.json"
```
