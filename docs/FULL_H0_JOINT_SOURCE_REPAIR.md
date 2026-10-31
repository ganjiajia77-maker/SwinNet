# Full H0 continuation vs joint source repair

## Identity and scope

A and B both load the original Massachusetts H0 strictly from:
`/home/gjj/SwinNet-h0-mass/model_out/mass_h0_599a410_random512_fp32_80e_bs2_20261011_113054/best.pth`.
The supplied inventory records epoch 65, 1010 tensors, no EMA, high-resolution stream,
H3 surface fusion and global topology enabled, Stage2/3 weights .008/.012.

`reference_h0_599a410/` isolates the ORIGINAL network, loss and direction-target code
from commit 599a410de8d341850e1810417307e4b3bbdfc82d. Only absolute import paths were
namespaced. SOURCE.json records the original and packaged hashes. The original
criterion builders were extracted from that commit without importing its CLI.
No current local-gap architecture or permissive partial checkpoint loading is used.
Existing detach/gradient ratios INTERNAL to H0 remain as originally configured;
only the external frozen BASE/Q3 boundary is removed.

B loads the ACTUAL joint-transport checkpoint from
`h0_v9_source_selector_20261030_110206/actual_transport_20261030_181042`.
Version, experiment configuration, completion best score and reported own-val .67
work point are checked. Each used repair module and Q3 head loads strictly. The
old local-gap BASE is explicitly REPLACED, never loaded as the complete H0.
The exact Q3 checkpoint is read from source identity and SHA256 checked, not guessed.
The old standalone Q3 R3 residual heads and scalar state_gate are not used in this
source-selector forward. Scalar-G-only parameters are excluded from optimization.

## Forward and losses

A: full H0 -> original P0 and auxiliary predictions -> original H0 loss.

B: same full H0 -> LIVE H2, Stage3 refined decoder feature, P0, S2/C2 -> trained
Q3 -> original visual/R/W/B/A/rates, projected pair features and source selector ->
three eight-neighbour updates -> original logit-increment surface composition.
The implementation shares the source-selected update and proposal equations with
the validated source repair. It does NOT use the slim segmentation backbone.

Transport = rate_transport * sum(eight choice weights * (neighbour U - current U)).
Stay is the ninth choice. Proposal=.2*A*rate_proposal*(B-U), anchor=rate_anchor*(P0-U).
The next step reads the NEW preceding state. Invalid neighbours have zero choice.
No extra scalar G multiplies transport.

B loss: original H0 loss + original S_selector Lseg + .2Lcont + original configured
lambda_relation*Lrelation (normally .1). Q3 learns through final prediction, with
no newly added independent Q objective. There is no L_choice, transport-benefit
L_G, candidate path loss or shadow/texture augmentation.

## Initialization and common training protocol

All used trained repair weights are loaded. To start near identity, ONLY the two
existing rate-head biases for transport and proposal receive -8. This is recorded
initialization, not a new formula or gate. Smoke measures initial |P1-P0|; requires
mean <= .001 and maximum <= .02, otherwise stops before training. It also records
threshold-crossing pixels and direct P1 segmentation gradients to H0/Q3/selector.

Default BOTH groups: 80 ADDITIONAL epochs, batch2, FP32, accumulation1, no EMA,
seed1234, native1500 random512, original geometry/color augmentations, four workers.
H0 and Q LR=5e-5 ->5e-6, 10-epoch warmup and cosine. Existing repair groups use
their S_selector learning rates with the same warmup/cosine multiplier and .1 LR
floor. AdamW weight decay .0001. H0 LR and schedule match exactly across A/B.
This continuation uses fresh optimizer state; own last.pth restores optimizer/RNG.
The data order, crop/augmentation epoch and per-batch H0 stochastic seed are equal.
After BOTH finish, comparison checks order and augmented RGB tensor hashes for
EVERY epoch. An old B_control or slim run cannot substitute for this new A.

## Validation

The first process computes a shared initial H0 P0@.45 val reference BEFORE training.
A file lock prevents two concurrent reference passes. The second process waits
only for this initial pass; A/B training then runs concurrently on separate GPUs.
These initial complete-H0 gaps/anchors differ from the old local-gap 206 set.
They are never regenerated from a fine-tuned P0.

Checkpoints: val@.60 every five epochs and final epoch, identical rule:
IoU + .15clDice + .10APLS_GT_to_Pred + .05APLS_bidirectional
- .001short_pred_components - .01background_bridge_proxy/images.
Final validation thresholds: .35 to .75 step .01. Reports include each own selected
threshold, common A threshold, exact matched Recall and matched foreground area,
plus B's underlying P0 at the common threshold. Matching residual counts are saved.
The last finishing group automatically creates comparison/REPORT.md and CSV.
APLS, fixed-gap closure and background bridges remain proxy metrics. No test tuning.

## Commands and artifacts

`GPU=1 EPOCHS=80 bash scripts/run_full_h0_joint.sh A all`

`GPU=2 EPOCHS=80 bash scripts/run_full_h0_joint.sh B all`

Both default to `/home/gjj/SwinNet-roadbias/model_out/full_h0_joint_pair_v1`.
Use the SAME explicitly set PAIR_DIR to create a separate pair. Running the same
command resumes its last.pth; completed training proceeds to val. Each variant has
its own lock; the pair contract rejects mismatched budget/seed/LR/dataset.

Key outputs: reference/fixed_gaps_and_anchors.json, A|B/experiment.json,
A|B/smoke.json, A|B/epoch_*.json, best.pth, last.pth, eval/val/*.npz,
eval/val_selection.json, comparison/REPORT.md, comparison/comparison.csv.

Local integration tests use the REAL original H0 at 128x128 and the actual repair
modules with synthetic weights/data. Real server checkpoint strict loading and
512x512 physical-batch2 CUDA memory/time checks run before formal training. Local
tests do not establish Massachusetts accuracy or GPU memory consumption.
