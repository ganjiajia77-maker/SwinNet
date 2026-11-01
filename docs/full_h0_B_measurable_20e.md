# B with measurable initial rate budgets; reuse completed A

Run `bash scripts/run_full_h0_B_measurable_20e.sh` with `GPU=1`.
The default pair is `full_h0_joint_pair_20e_20261101_085322`.

- Read the completed `A` control; verify 20 epochs, settings, original H0
  implementation hashes, best checkpoint SHA and all 14 validation predictions.
- Rebuild B from the same starting full H0 and validated actual-transport
  S_selector checkpoints. Do not continue from the previous tiny-rate B or
  from weights overfitted by the 200-update startup diagnostic.
- Initialize the final four-way rates output with zero weights and logits
  giving transport=0.01 and proposal=0.001. Stay/anchor retain their previous
  ratio. The effective proposal coefficient remains `0.2*A*proposal_rate`.
- Keep complete H0/Q, all existing source selection and repair heads, eight
  neighbours, three steps, optimizer ranges, schedule and losses unchanged.
- Store the new model in `B_transport_0p01_proposal_0p001`; preserve old A/B
  weights, predictions and comparison reports.
- Verify each epoch's input order and augmented RGB fingerprints against A.
  Resume only the new B's strictly matching checkpoint/configuration.

The legacy 1e-7-budget initialization used a near-identity max-pixel guard.
This explicitly different startup setting uses the budgets already examined
in the short audit. Record its real initial mean/max P1-P0 and binary flips;
verify finite nonzero transport, selector influence and segmentation gradients.
Do not describe it as an exactly identity initialization or a proven improvement.

Training logs and epoch JSON record rate and actual transport, proposal/anchor
increments, P1-P0, and gross TP/FP additions/removals at threshold 0.45. Each
validation also records P1/current-P0 changes at 0.60 without changing selection.
Checkpoint and threshold selection retain the original validation-only rule.

After the 20 epochs and validation threshold sweep, automatically compare the
new B with saved A at own/common thresholds, matched Recall and matched area.
Write `comparison_B_transport_0p01_proposal_0p001/REPORT.md` and CSV.
APLS, gap closure and bridges retain the existing proxy definitions.
