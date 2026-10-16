# e2020d9 surface-only Massachusetts ablation

Base: `e2020d9a8df311f9f680cd3637466a674d76688e`.
Compare with the full e202 Mass protocol, not the H0 or Conv-C variants.

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
All surviving state tensors are exactly equal to the same-seed full model
before training. The original ImageNet loading rule and loaded/new LR groups
are retained. Both load 225 compatible ImageNet tensors at 512 input.

Only focal BCE (`gamma=1`) plus `0.5 * Dice` supervises the final surface.
No skeleton, connectivity, direction, boundary, coarse-mask or topology loss
is applied. The dataset bypasses thinning; zero skeleton placeholders retain
the original trainer batch contract. Image/mask augmentation and random crop
coordinates remain identical to the full-model dataset for each seed/epoch.

The common surface head has 124,945 parameters in both variants. At 512 input,
the full model has 77,013,094 parameters and this ablation has 74,354,667
(2,658,427 fewer). No unused layers are added to match total parameter count.
The unchanged reference DCA contains a weight predictor whose intermediate
result is overwritten in its original forward; it is retained in both arms.

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
cd /home/gjj/SwinNet-e202-surface-only-mass1500
export CUDA_VISIBLE_DEVICES=1
export SEED=1234
export RUN=e202_surface_only_mass1500_random512_fp32_noema80_b2_acc1_s${SEED}_$(date +%Y%m%d_%H%M%S)
bash scripts/run_e202_surface_only_mass1500.sh train
bash scripts/run_e202_surface_only_mass1500.sh sweep
bash scripts/run_e202_surface_only_mass1500.sh test
bash scripts/run_e202_surface_only_mass1500.sh topology
```

`test` already computes topology and writes `test/test_comparison.csv`.
`topology` reuses the completed report. Set RUN to the existing run when
resuming, then use `bash scripts/run_e202_surface_only_mass1500.sh resume`.
Independent seeds should also be run for the full reference arm.

```bash
python -u scripts/benchmark_surface_only.py \
  --pretrain_ckpt /home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth \
  --output_json /home/gjj/SwinNet-roadbias/model_out/$RUN/full_vs_surface_compute.json
```

The benchmark measures full forward, including every enabled model branch,
on the same GPU with batch2/512/FP32. It reports parameter counts, shared-head
counts, synchronized batch/per-tile time and peak allocated GPU memory.
It excludes H2D, sliding fusion and metrics; it does not measure full-image
inference or training epoch time. Report measured server results for speed.
