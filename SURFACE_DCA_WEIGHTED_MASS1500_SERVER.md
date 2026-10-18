# e2020d9 surface-only weighted-DCA Massachusetts experiment

Base: `e2020d9a8df311f9f680cd3637466a674d76688e`.
Parent: `eb2da24d768b7f643e4fb288e3736693862c1fe8`, the surface-only Mass run.
Compare this experiment with that surface-only baseline.

## DCA change

Both DCA skip modules now compute
`fusion_conv(cat(K * softmax(weight_conv(Q))[k] * sampled[k]))`.
The old weighted sum that was overwritten is removed. The weight predictor's
last convolution starts at zero, so every initial multiplier is one. All other
parameters and random-number consumption retain the original initialization;
the initial function equals the original unweighted concatenation. On the first
step the final weight-prediction layer receives gradients; its preceding layer
receives nonzero gradients after the final layer starts learning.

Parameter count stays 74,354,667; the existing weight predictor becomes active.
This is a new from-scratch experiment with the original ImageNet pretraining.
Do not resume the completed unweighted run as this experiment. Both DCA blocks
save `weighted_fusion_version=1`; strict loading and manifest validation reject
old checkpoints. Measured server epoch/forward timing is needed for any speed claim.

## Effective architecture

Kept: original edge PatchEmbed, encoder depths `[2,2,6,2]`, PatchMerging
(linear plus MLP reduction), global-local bottleneck and its Swin block,
decoder Swin depths `[6,2,2]`, all three skip paths and both DCA modules,
FinalPatchExpand, and the original surface projection, branch, refinement and head.
The YAML `DECODER_DEPTHS` field is not used by this e202 constructor; these are
the actual block counts.

Removed: A1/A2 road heads and their downsample/attention guidance, Stage 2/3
S/C/D predictions, structure gates/residuals and H2-to-H3 transfer, the
prepatch z_struct/E128 stream, all high-resolution feedback, H3-to-surface
fusion, global topology and final boundary prediction/residual.

`--surface_only` builds the reference configuration on CPU and prunes these
modules after initialization, before moving to GPU or creating the optimizer.
The runner retains the full-model constructor options only to preserve the RNG
sequence. They do not enable the pruned runtime branches. Checkpoint
`topology_coefficients` records the effective disabled state.
All shared parameters except the last DCA weight-prediction layers keep the
same-seed baseline values. The original ImageNet loading rule and loaded/new LR groups
are retained. Both load 225 compatible ImageNet tensors at 512 input.

Only focal BCE (`gamma=1`) plus `0.5 * Dice` supervises the final surface.
No skeleton, connectivity, direction, boundary, coarse-mask or topology loss
is applied. The dataset bypasses thinning; zero skeleton placeholders retain
the original trainer batch contract. Image/mask augmentation and random crop
coordinates remain identical to the full-model dataset for each seed/epoch.

The common surface head has 124,945 parameters in both variants. At 512 input,
the full model has 77,013,094 parameters and this ablation has 74,354,667
(2,658,427 fewer). No unused layers are added to match total parameter count.
Both DCA weight predictors now participate in the loss and output.

## Protocol

Native 1500 images under `/home/gjj/Swin-Unet-main/Massachusetts/tiff`, with
`train`, `val`, `test` and matching `*_labels` directories. Each training
image yields one random 512 crop per epoch. Physical batch 2, accumulation 1,
80 epochs, seed 1234, FP32, no EMA; no 3000-batch limit.
Validation every 5 epochs. Pretrained LR `5e-5 -> 5e-6`, new LR
`2e-4 -> 1e-5`, LR warmup 10 epochs, original AdamW and clipping.

Validation/test: native-resolution 512 tiles, stride 256, original taper0.1
weighted-logit stitching then sigmoid. No TTA or postprocessing. Threshold
selection uses only validation global IoU over 0.05 through 0.95 in 0.05 steps.
Test reports global segmentation counts plus per-image topology using the
same versioned road-mask-v1.1 tool as the full e202 Mass run. APLS columns are
the shared raster approximation, not official SpaceNet APLS. Evaluation uses
1500 resolution, short area 20, max APLS nodes 64, snap radius 5 pixels.

## Server

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
cd /home/gjj/SwinNet-e202-surface-dca-weighted-mass1500
export CUDA_VISIBLE_DEVICES=1
export SEED=1234
export RUN=e202_surface_dca_weighted_mass1500_random512_fp32_noema80_b2_acc1_s${SEED}_$(date +%Y%m%d_%H%M%S)
bash scripts/run_e202_surface_dca_weighted_mass1500.sh train
bash scripts/run_e202_surface_dca_weighted_mass1500.sh sweep
bash scripts/run_e202_surface_dca_weighted_mass1500.sh test
bash scripts/run_e202_surface_dca_weighted_mass1500.sh topology
```

`test` already computes topology and writes `test/test_comparison.csv`.
`topology` reuses the completed report. Set RUN to the existing run when
resuming, then use `bash scripts/run_e202_surface_dca_weighted_mass1500.sh resume`.
Independent seeds should also be run for the surface-only reference arm.

```bash
RESULTS=/home/gjj/SwinNet-roadbias/model_out
RUN=$(cat "$RESULTS/latest_e202_surface_dca_weighted_mass1500_run.txt")
AUDIT=$(cat "$RESULTS/$RUN/latest_unified_eval.txt")
cat "$AUDIT/test/test_comparison.csv"
```

The runtime package includes training, inference and unified evaluation only.
The original surface-only latest-run pointer and results are separate.
