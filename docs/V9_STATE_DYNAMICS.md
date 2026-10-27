# V9 state-aware staged dynamics

This branch starts from original V8 `cf87c33`. Its transport coefficient is still
`alpha_k = rates_k[:,1] * sigmoid(old_gate_logit + relation_delta + state_delta_k)`.
It does not include the separate V8 experiment multiplying alpha by V.

The state gate is shared by all three steps: 6 -> 16 -> 1, GELU, 129 parameters.
Its last convolution is zero initialized. Inputs are U, M-U, V, transport rate,
B-U and P0-U, computed in that order after source aggregation and rate prediction.
The frozen BASE/Q3 outputs are detached; no GT is used in model forward.

## Execution prerequisites

`preflight` checks V7/V8 saved module names, versions and source paths. V7 stores
the six downstream modules and pointers to BASE/Q3. V8 stores gate_relation and
a pointer to V7. Neither contains the missing frozen BASE/Q3 weights.

Existing reports and PNG panels are not complete upstream feature caches.
No complete feature-cache manifest was supplied for this task. The existing
`frozen_override` tensor interface is preserved and tested, but this runner loads
the full model from checkpoints. It does not claim to train from PNGs or fixed
R/W/B/A caches. Restoring original weights or rebuilding a consistent baseline
is required before full-model training.

For a rebuilt BASE/Q3, `scripts/run_v9_mass.sh bootstrap` trains a fresh V7 for
20 epochs and the original V8 gate for 20 epochs. Q3's own metadata chooses the
exact frozen BASE path, including when Q3 used an intermediate base snapshot.
This is a new baseline and cannot reproduce the deleted original experiment.
All subsequent A/B/C/D comparisons use that same baseline and frozen inputs.

## Stages and learning rates

| Stage | Trainable modules | Starting point |
|---|---|---|
| A | None; original V8 | V7 + V8 + BASE + Q3 |
| B | gate_relation, state_gate | A |
| C | B plus R, DirectionRelations/W, rates | B best |
| D | C plus visual, proposal B, acceptance A | C best |

BASE/Q3 always remain frozen in eval mode. Learning rates: state_gate 1e-4,
gate_relation 5e-5, R/W/rates 1e-5, visual/B/A 5e-6. CLI arguments can override
these initial experimental values. AdamW receives each trainable parameter once.
The loss remains the source checkpoint's L_seg + lambda_cont*L_cont, with the
existing focal/Dice and missed-skeleton continuity implementations.

Stage 0 checks every validation sliding window, comparing G, all three alpha/U
states and P1, and compares stitched full-image IoU. It must pass before training.
The real-data `smoke` command takes two optimizer steps per stage, checks finite
outputs and nonzero module gradients, and restores every downstream weight.
Both reports are required by the training command.

Each stage writes independent best/last checkpoints. Last includes the optimizer,
CPU/CUDA/NumPy RNG, learning rates, data/loss configuration and exact SHA256 source
checkpoint identities. Resume uses the same stage and total --epochs. Checkpoints
contain all downstream modules, gate_relation and state_gate, not frozen BASE/Q3;
keep those upstream files. Full validation selects checkpoints at threshold 0.60.

The default schedule adds 20 epochs in each B/C/D stage, so cumulative training
budgets differ. These stage comparisons do not isolate unfreezing from extra
training time; report the schedule when interpreting results.

## Metrics and diagnostics

`compare` evaluates only val, using a 0.35-0.75 threshold grid with step 0.01.
It saves own selected thresholds, a common threshold, matched Recall and matched
foreground area against V8. Matching uses 10,000-bin global validation probability
histograms (resolution 0.0001), then recomputes metrics at the selected thresholds.
Actual match errors are included; ties can still prevent exact equality. P0 reference threshold is fixed
for gap definitions, added/removed pixels and all stage comparisons.

The report includes IoU/F1/P/R, clDice, both directional and bidirectional raster
APLS, short components, added foreground/background, gap closure and gap-length
bins. APLS is approximate. Closure and background bridge are diagnostic proxies.

`audit` uses all validation images and sliding windows. Regions are defined from
stitched full-image predictions: recovered GT foreground, unrecovered GT foreground,
Q3-high background, isolated newly added FP and newly added FP near multiple old
components. The last category is a bridge proxy, not an authoritative GT edge label.
Masks are projected to uniquely owned working-grid samples; the internal maps are
tile-level values, not a reconstruction of a unique full-image latent state.
Signed/absolute means and histogram-approximate quantiles are reported per step
for V, M-U, G, rates and actual transport/proposal/anchor updates. A high-G,
near-zero-transport fraction, W entropy and positive/negative proposal outputs
are included. Audit thresholds default to the Recall-matched validation points.

Training logs include module gradients, first-batch L_cont gradients, missing
segments/path counts, B saturation, R/G extremes and three-step update summaries.
No model performance improvement is claimed before real validation results.

## Commands

Activate the existing swinunet environment. Set `GPU`, `V9_RUN`, and either the
original `V7_CKPT`/`V8_CKPT` or the rebuilt Q3 checkpoint. The shell helper records
V9_RUN in `model_out/latest_v9_run.txt` for use from another terminal.

```bash
export GPU=1
export V9_RUN=/home/gjj/SwinNet-roadbias/model_out/h0_v9_$(date +%Y%m%d_%H%M%S)

# If rebuilding: Q3 must have finished, and its referenced BASE must exist.
export Q3_CKPT="$(cat /home/gjj/SwinNet-roadbias/model_out/rebuild_q3_r3_run.txt)/best.pth"
bash scripts/run_v9_mass.sh bootstrap

# After either original dependencies or the rebuilt baseline are available:
bash scripts/run_v9_mass.sh preflight
bash scripts/run_v9_mass.sh stage0
bash scripts/run_v9_mass.sh smoke
bash scripts/run_v9_mass.sh b
bash scripts/run_v9_mass.sh c
bash scripts/run_v9_mass.sh d
bash scripts/run_v9_mass.sh compare
bash scripts/run_v9_mass.sh audit

# Or execute all post-baseline steps, stopping on any failure:
# bash scripts/run_v9_mass.sh all

# Example resume:
# bash scripts/run_v9_mass.sh resume_c
```

CPU checks: `python checks/test_v9_state_gate.py -v`. They use real downstream
PyTorch modules with a synthetic frozen feature provider, verifying exact zero-init
identity, state input order/sharing, Surface sensitivity, optimizer coverage,
module gradients/updates, frozen parameters, strict checkpoint restoration,
feature override equivalence and missing-upstream rejection. They do not measure
Massachusetts accuracy, GPU memory or actual-checkpoint numerical identity.
