# Slim end-to-end source-selected road repair

This is a new whole-system experiment, **not a single-module ablation of frozen
V9 or S_selector**. No previous BASE, Q3, V7 or V9 checkpoint is loaded. Only the
same ImageNet Swin initialization is used for fresh training. Its loaded tensor
names and SHA256 are recorded. New checkpoints contain the entire segmentation,
Q and repair model; inference/resume no longer depends on separate upstream files.

## Actual feature path for 512 input

| Component | Tensor / role |
|---|---|
| Pure segmentation encoder | Existing PatchEmbed/CNN, 4 Swin stages [2,2,6,2], existing downsampling/bottleneck |
| Pure segmentation decoder | Existing upsampling, [6,2,2] Swin blocks, skip concat and two DCA blocks |
| Semantic input | `layers_up[1]` output: B,192,64,64 |
| Detail input | Final `norm_up` output: B,96,128,128; already includes refined skips |
| P0 | Existing final x4 expansion + original surface_proj/branch/refine/head: B,1,512,512 |
| Feature adapters | Detail 96->64 at128; semantic 192->96 at64, aligned to128 |
| Rebuilt Q | `[detail64, semantic96, P0_1]` -> existing lightweight multiscale32 -> Q1 at128 |
| R/W | Existing V9 reliability and direction-relation modules; reinitialized and trained |
| Source relation context | Existing local/context/8 pair features; scalar DeltaG parameters removed |
| Selector | Actual eight valid neighbours + stay, state-dependent at every step |
| Three-step update | `rate * sum(a_source*(U_source-U)) + .2*A*rate_proposal*(B-U) + rate_anchor*(P0-U)` |
| P1 | `sigmoid(Z0 + upsample(logit(U3)-logit(P0_work)))` |

Old S/C/D heads, auxiliary losses, predicted structure feedback, A1/A2 attention,
independent H0 RGB stream, H<->G interaction, global topology and old Gap path are
pruned. The old constructors run transiently to preserve common backbone module
initialization order, then these modules are actually removed before any forward
or optimizer exists. Old constructor log lines describe that transient assembly;
the `[SLIM]` line and parameter inventory describe the resulting model.

No DCA sampling-weight repair is included: the existing fusion output is retained.
The old local-gap implementation already removed its unused weight_conv branch.
There is no new independent RGB encoder and no S/C/D input replacement with zeros.
The reused W helper still returns an unused scalar compatibility value, but its
scalar head row has no parameters; it never enters the actual source update.
The source context helper's old DeltaG is a parameterless zero operation. The
new model does not compute/discard a separate old scalar-G dynamics loop.

## Gradients and training

The new forward bypasses all old frozen wrappers. No no_grad/detach/forced eval
surrounds segmentation features, P0, Q or repair. Encoder/decoder BN, dropout and
Swin stochastic depth follow train mode. GT only generates loss targets,
eligibility and matching under no_grad; forward accepts only the image.

Fixed initial objective:

    L = Lseg(P1) + 0.2 Lcont
        + 1.0 Lsurface(P0) + 0.1 LQ
        + 0.1 Lrelation + 0.05 Lchoice

P0 uses FocalBCE(gamma1)+0.5Dice; P1 retains V9's weighted FocalBCE(gamma2)+0.5Dice.
Q retains its missed-skeleton proxy definition: tolerance2px around detached
current P0, probability0.5, minimum3px components, dilation2px and max-pool to its
output scale; balanced BCE+0.5Dice. Since P0 trains, targets are refreshed online
with stopped gradients, rather than read from the old frozen Q3 model.
Skeleton omission is a proxy, not an exact topological break annotation.

Native relation labels, ignore/matching and three-step relation BCE are reused.
Actual wrong-choice LSE loss is explicitly **on**, initial coefficient0.05, at
steps2/3 only. This is an unvalidated setting fixed before training, not a claim
that the preceding frozen experiment succeeded. No old L_G or candidate path
losses are included. Remaining losses/coefficients are written to experiment.json.

Mass1500 native -> one random512 crop/image/epoch; val/test512 stride256 logit
blending; physicalbatch2, accumulation1, FP32, noEMA, seed1234, workers4, 80epochs.
ImageNet-loaded parameters: LR5e-5 ->5e-6; other parameters2e-4 ->1e-5. Same
10epoch warmup and cosine schedule, AdamW(weight_decay.01). Validation every5
epochs; selected checkpoints use unchanged V9 fixed val@0.60 score:

    IoU + .15clDice + .10APLS_GT_to_Pred + .05APLS_bidirectional
        - .001short_components - .01bridge_proxy/images

`ACTIVATION_CHECKPOINT=1` optionally recomputes Swin blocks to reduce memory;
default0 preserves the previous configuration. The actual choice is fixed in
the run metadata. No formal training is started on the local machine.

## Commands

    bash scripts/run_slim_end_to_end_mass.sh all
    bash scripts/run_slim_end_to_end_mass.sh resume
    bash scripts/run_slim_end_to_end_mass.sh sweep
    bash scripts/run_slim_end_to_end_mass.sh test
    bash scripts/run_slim_end_to_end_mass.sh topology
    bash scripts/run_slim_end_to_end_mass.sh results

`all` performs one real batch check, training and val threshold selection. Test
is a separate explicit final evaluation, never a model-selection input. The script
records the absolute run path in `latest_slim_end_to_end_mass_run.txt`.
Existing `last.pth` resumes under `all`/`resume`; `train` starts a new run by default.

The startup check reports actual512 batch2 shapes, module/encoder gradients,
P1-to-encoder gradients, runtime and CUDA peak memory, and restores BN buffers so
the check cannot alter the training initialization. Each checkpoint loads strictly;
last includes optimizer and Python/NumPy/Torch/CUDA RNG. Only best/last are retained.

## Evaluation boundary

Saved P0/P1 probabilities and CSVs compare own val thresholds, common threshold,
matched Recall and matched foreground area. Own val thresholds use the same
Mass-control grid0.05..0.95 step0.05; Recall/area matching uses actual probability
order statistics with tie errors reported. Test uses exclusively val thresholds.

Because this model's P0 is trainable, the old frozen V9 206-gap manifest is not
reused as if it described this baseline. The selected model's P0@0.45 reference
gaps/anchors are fixed once per split and held identical across P0/P1 thresholds.
That reference, its fingerprint and length-bin closure are exported; counts are
not directly comparable to old206-gap counts. APLS/closure/bridge are proxies.

Inspect `smoke.json`, `parameters.json`, `epoch_*.json`, `eval/val/comparison.csv`,
`eval/test/comparison.csv`, `eval/*/summary.json` and `eval/*/*.npz`.
The new architecture removes several components together; overall performance
cannot identify the individual contribution of deleting any one of them.

## Local verification

Five tests passed using the real Swin backbone, batch2 at128 input: removed
modules/retained block counts, live encoder/decoder/Q/R/W/B/A/rates/selector
gradients, real-batch smoke with BN restoration, strict self-contained checkpoint
serialization, LR schedule and area matching. A separate actual512 CPU forward
confirmed detail96x128x128, semantic192x64x64, Q1x128x128 and final1x512x512.
Full model: **74,270,754 trainable parameters**. These are local synthetic input
checks, not a Mass training result or CUDA measurement. Actual512 batch2
forward/backward memory/time is measured by the server startup smoke.
