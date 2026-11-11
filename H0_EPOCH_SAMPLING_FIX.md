# H0 encoder post-merge experiment: sampling repair

This branch starts at evaluation commit `0a8e685` (training model `7d2bc9b`).
Network files, losses, model initialization, optimizer and scheduler are unchanged.
The old result directory/checkpoint is retained and must not be resumed here.

## Restored data implementation

`datasets/dataset_road_skeleton.py` and `mass_protocol.py` are copied exactly from
the original Massachusetts H0 package `fe3b730`. This restores:

- shared epoch visible to DataLoader processes;
- original deterministic randomized affine traversal of crop coordinates;
- synchronized image/road mask/skeleton processing and original augmentation;
- native size, case pairing and binary GT checks, accepting 0/1 and 0/255 masks.

For native1500/random512, every image has 80 distinct crop positions over epochs
0..79. A seed makes the sequence reproducible; sampling is not based on predictions
or road density. Geometry/color augmentation remains unchanged.

At the user's request, persistent workers are disabled by default in this branch,
including after reference checkpoint configuration inheritance. Four workers are
still used, but recreated each epoch. `h0_reference_config.json` records this
explicit override rather than incorrectly claiming all worker settings match H0.
This policy can alter the global shuffled visitation order through worker-seed
generation. Equality here means the same image/crop/epoch has the original H0's
RGB and targets, not a claim of identical historical minibatch order.

## Checks before fresh training

Run in the `swinunet` environment, with a new `PHASE_RUN_NAME`:

```bash
bash scripts/run_h0_encoder_phase.sh check
bash scripts/run_h0_encoder_phase.sh smoke
bash scripts/run_h0_encoder_phase.sh train
```

`check` runs only on CPU and performs no optimizer steps. It checks normalized
source hashes against the original H0 data implementation, verifies all training
image indices have 80 distinct crop coordinates, then compares multiprocess
RGB/road/skeleton outputs elementwise with direct dataset access for the same
indices in epochs 0, 1 and 79. It saves `${PHASE_RUN_NAME}_sampling.json` next to
the run directory. `smoke` checks the 512/batch2 forward and task gradients without
optimizer steps. Fresh training requires both checks to pass with matching seed,
worker policy, initialization and original H0 reference identity.

Only one model is trained, from the same SwinV2/ImageNet initialization, for 80 full
epochs. Batch2, FP32, no EMA, no batch cap. The reference H0 road checkpoint supplies
configuration only. Neither the failed experiment nor its optimizer is loaded.
`resume` rejects old checkpoints lacking the corrected sampling protocol marker;
it may resume only an interrupted run of this repaired version.

Evidence:

- `${PHASE_RUN_NAME}_sampling.json`: source identity and per-image/epoch crop checks.
- `h0_reference_config.json`: inherited settings and explicit worker override.
- `sampling_epochs.jsonl`: first minibatch coordinates and dataset epoch each epoch.
- `phase_initial_state.json`, `_smoke/phase_smoke.json`: original initialization/gradient checks.

The evaluator `scripts/evaluate_h0_encoder_phase.py` is unchanged: select threshold
on validation, infer test using the fixed threshold, then six topology metrics with
APLS control cap150, TOPO interval0.002/step0.0002, TLTS200, seed1234.
This fixes experimental validity; it does not guarantee an accuracy improvement.
