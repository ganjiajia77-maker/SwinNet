# Late wrong-source choice mass experiment

The original source forward, native relation labels, matching rule, joint training
files and code hashes remain unchanged. New files implement the requested loss
and a continuation from the **selected current actual-transport S_selector**.
No relation pretraining or B_control training is repeated.

`networks/v9_late_choice_loss.py` reads the live stay/eight edge logits. It masks
invalid sources and computes, at steps 2 and 3 only:

    logsumexp(all valid nine choices)
      - logsumexp(stay + sources not confirmed wrong)

GT labels and eligibility are detached; choice logits are live. Unknown sources
are allowed. Existing road-evidence eligibility is converted to a position mask;
at each eligible position ALL confirmed-wrong valid neighbours are penalized,
including edges not selected by class matching. All-wrong neighbourhoods remain
supervised. Eligible positions with no wrong source contribute zero. Losses are
averaged over positions, then nonempty images, then equally over steps 2 and 3.
An entirely empty step contributes graph-connected zero and is reported. A whole
epoch without wrong-bearing positions or matched relation labels aborts.

The unchanged relation loss still supervises all three steps. The full loss is
`Lseg + 0.2 Lcont + 0.1 Lrelation + 0.05 Lchoice`. The last coefficient is an
unvalidated initial setting. Step 1 has no **direct** choice loss; joint training
and gradients through later states can still change its behavior.

`run_v9_late_choice.py` strictly loads the current joint checkpoint, verifies its
original code hashes, frozen BASE/Q3 identity, fixed 206 gaps/anchors and existing
loss coefficients. It reuses exactly the joint parameter groups and learning
rates. BASE/Q3 and the unused scalar G heads stay frozen. Best checkpoints have
no optimizer; continuation starts fresh AdamW with the same LR/weight decay,
and its own `last.pth` restores optimizer/RNG exactly. Crop epoch advances from
the selected source epoch. Default continuation budget: 20 epochs.

On the server:

    bash scripts/run_v9_late_choice.sh all
    bash scripts/run_v9_late_choice.sh results

The script resolves the source through the known selector run's
`latest_actual_transport_run.txt`. It creates a separate run, leaves the source
best untouched, runs one real batch check, trains, then evaluates on val only.
An existing run resumes from its own last checkpoint. GPU defaults to physical 1.

Outputs:

- `choice_smoke.json`: actual softmax equality, selector/stay gradients, frozen
  backbone and weighted-choice/segmentation gradient ratio; no optimizer step.
- `epoch_*_choice_positions.json`: each batch/image/step's supervised positions,
  skipped images, all-wrong neighbourhoods, wrong mass, choice loss.
- `actual_source_relations.csv`: actual supported/blocked source weights and
  signed/positive transport by step, before and after continuation.
- `comparison.csv`: current S, its stop_late contrast, and continuation at the
  common threshold; continuation's own val threshold and matched Recall.
- `late_road_retention.csv`: GT roads retained and FP beyond current stop_late.
- `val_probabilities_*.npz`, frozen reference copies, `REPORT.md`.

Checkpoint selection retains the original fixed val@0.60 score:
`IoU + .15 clDice + .10 APLS_GT_to_Pred + .05 APLS_bidirectional
 - .001 short_components - .01 bridge_proxy/images`.
Final val threshold grid remains 0.35–0.75 in 0.01 increments. Matching Recall
uses actual positive probability order statistics, and reports unavoidable ties.

Success requires less wrong positive transport/FP/bridging while retaining true
late recovery; merely converging to stop_late is insufficient. The historical
38,128 recovered roads / 60,667 FP are context, never fabricated outputs. Extra
training is a confound: a pure loss-effect claim needs an equal-budget
`lambda_choice=0` continuation from this same source with identical crop/order/LR.
This optional control is supported but is NOT launched automatically.
APLS, closure and bridge results remain proxy metrics. No test tuning occurs.

Local checks use real downstream modules with a synthetic frozen feature source.
Full Mass/CUDA and scientific outcome checks run on the server; no validation
improvement is claimed before those results exist.
