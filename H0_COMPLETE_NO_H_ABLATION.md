# Complete-no-H ablation on ccc7992

This independent experiment removes the entire H stream and the final H-driven
global topology branch. It retains the corrected original Mass H0 data protocol.
Only one new model is trained from SwinV2/ImageNet initialization for 80 complete
epochs, batch2, FP32, no EMA. No road checkpoint/optimizer is used as a warm start.

## Exact scope

Removed: RGB H stem, H adapter and auxiliary head, all three E/H exchanges,
H-specific decoder refinement and bidirectional projections, H-only surface
correction modules, and the final `global_topology` sparse-anchor module.
A1/A2 stay removed as in ccc7992. No replacement global module is introduced.

Retained: encoder/merging/skip paths, bottleneck context and Swin block, Stage3 GAP
calibration, G2/G3 and their existing inter-stage fusion, S/C/D prediction and loss,
G-only convolution/activation inside the old bidirectional fusion blocks, G3
surface projection/fusion, surface and boundary heads. The original S/C/D and
surface loss implementations and effective weights are unchanged. The removed
H-specific auxiliary loss already had zero weight in the original reference.

The ordinary original-H0 architecture is constructed and ImageNet weights loaded
before the ccc7992 phase baseline is installed. H-only modules are then removed.
Every surviving state tensor must equal its baseline value, and no new parameter
or random draw is introduced by removal. The G-only wrapper reuses the same
`stage_refine` convolution and activation; it does not bypass that computation.
Old state-key prefixes containing `highres_skeleton_fusion.stage_refine` remain
for these retained G-only parameters. The prefix is not evidence of an H stream.

Checkpoint args retain original H0 *construction* flags so loading can reproduce
the original initialization and module names. `complete_no_h_ablation=True` then
determines the effective architecture, which is explicitly recorded in
`h0_reference_config.json` and tested in the smoke/audit reports. The effective
core flags for H and final H-global are both false. This separates original
construction settings from the actual pruned forward graph.

## Run

In conda environment `swinunet`, set an independent `NO_H_RUN_NAME`, then:

```bash
bash scripts/run_h0_no_h.sh check
bash scripts/run_h0_no_h.sh smoke
bash scripts/run_h0_no_h.sh train
```

The wrapper pins the no-H variant. The first command checks all training image
crop sequences across 80 epochs and worker/direct RGB/GT/skeleton equality. The
second checks a real 512/batch2 forward plus original loss gradients without
optimizer steps. Training requires both checks and identical smoke/fresh initial
states. Interrupted runs of this exact variant may use `resume`; a baseline-H
checkpoint or a pre-repair sampling checkpoint is rejected.

The original H0 checkpoint is configuration-only and remains the same full H0
reference as ccc7992. Four workers are recreated each epoch; persistent workers
are disabled. Crops/augmentation are the repaired ordinary random512 protocol.
Optimizer, learning rates, loss coefficients, validation threshold selection,
epoch length and seed are inherited without changing the baseline budget.

## Evidence and evaluation

`phase_initial_state.json`: retained tensors, zero mismatches, removed keys and
architecture checks. `_smoke/phase_smoke.json`: H absent, G/S/C/D shapes and task
gradients, optimizer_steps=0. `no_h_training.jsonl`: G task gradients, live G shapes
and H/global absence. `sampling_epochs.jsonl`: epoch and first-batch crop records.
Use `scripts/show_h0_encoder_phase.py --run_dir ...` to print these without pandas.

`scripts/evaluate_h0_no_h.py` strictly loads this ablation and reuses the existing
validation-sweep/test/six-topology evaluator. It labels output `h0_complete_no_h`.
No TTA, native1500, windows512/stride256, original tapered-logit blend minimum0.1;
threshold selected on validation only. Six metric protocol stays control cap150,
TOPO interval0.002/step0.0002, TLTS200 and seed1234. To evaluate the H baseline use
the baseline evaluator/checkpoint in its original experiment directory.

This comparison measures the combined contribution of H and its dependent final
global path. It does not separately attribute encoder, decoder or global effects.
