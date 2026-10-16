# eb2da24 surface + independent skeleton auxiliary tasks

Branch: `codex/e202-skeleton-aux-mass1500`.
Direct parent baseline: `eb2da24d768b7f643e4fb288e3736693862c1fe8`, whose direct
parent is `e2020d9a8df311f9f680cd3637466a674d76688e`.
The running surface-only experiment and its branch are preserved.

## Architecture and loss

The road forward is the original surface-only forward. Encoder depths `[2,2,6,2]`,
actual decoder Swin depths `[6,2,2]`, PatchEmbed/PatchMerging, bottleneck, both
DCA skips, final upsampling and the entire surface head are unchanged.

At each of decoder stages 2 and 3, immediately after its original `layer_up`:

```text
F2 (96x128x128) ---------------------> original next decoder stage
  +-> scale_gradient(0.5) -> original structure_branch -> skeleton_head -> S2

F3 (96x128x128) ---------------------> original surface prediction
  +-> scale_gradient(0.5) -> original structure_branch -> skeleton_head -> S3
```

Dimensions above assume RGB512 input. The two heads own independent parameters.
Their features/logits are returned for supervision only; they do not change
decoder tokens, do not enter a gate, do not feed the other auxiliary head, and
do not fuse into surface prediction. A1/A2 road guidance, C/D heads, structure
gates/residuals, stage structure transfer, high-resolution/E128 branches,
boundary residual and global topology remain physically removed as in eb2da24.

```text
L = L_surface + 0.008 * original_soft_skeleton_loss(S2,T2)
              + 0.012 * original_soft_skeleton_loss(S3,T3)
```

The existing `losses/road_losses.py`, soft target generation, skeleton BCE+Dice,
and dataset skeleton generation implementation are used unchanged.
`skeleton_pos_weight=2.0`; stage loss factors remain 1.0.
The 0.5 factor applies to gradients entering the shared decoder, not to the
0.008/0.012 loss weights. No detached auxiliary inputs and no soft-clDice loss.

Construction retains the original structure/skeleton submodules before the
surface-only pruning step. It does not create newly randomized replacements.
The shared initialization and constructor RNG state exactly match the same-seed
pure baseline; both load the same 225 compatible ImageNet tensors at 512.
The original zero-initialized skeleton readouts are retained. Their first
backward can give zero gradients to earlier shared features; after the readout
updates, the auxiliary loss propagates into the encoder and decoder.

The runner keeps the baseline's constructor flags, including high-resolution
and topology flags, solely to reproduce its initialization sequence. They are
pruned before training. The runtime architecture, checkpoint weights and
`skeleton_aux_protocol` record the effective auxiliary-only configuration.

## Same training and validation protocol

| Setting | Value, matching the surface-only baseline |
|---|---|
| Data | `/home/gjj/Swin-Unet-main/Massachusetts/tiff` |
| Train input | One native512 random crop per native1500 image per epoch |
| Augmentation | Original synchronized geometry and image appearance augmentation |
| Crop seed | 1234 by default; epoch shared with persistent DataLoader workers |
| Training | 80 epochs, FP32, physical batch2, accumulation1, no EMA |
| Initialization | Same ImageNet `swinv2_tiny_patch4_window8_256.pth`; fresh experiment |
| Loaded parameter LR | 5e-5 → 5e-6 |
| New parameter LR | 2e-4 → 1e-5, original cosine scheduler, warmup10 |
| Optimizer/clipping | Original AdamW and clipping |
| Periodic validation | Every5 epochs, original surface F1 selects best.pth |
| Inference | Native1500, 512 tiles, stride256, original taper0.1 weighted logits then sigmoid |
| Threshold | Validation global IoU selects from 0.05…0.95 in steps0.05 |
| Test metrics | Same unified road-mask-v1.1 tool as the pure baseline |

Both arms start separately from ImageNet plus identical shared random tensors.
`warm_start_ckpt` and `no_pretrain` are rejected for this auxiliary comparison.
`resume` is only for continuing this new experiment's own last.pth, never the
trained surface-only checkpoint. Strict checkpoint loading rejects mismatched
head sets. The new run prefix and latest pointer do not reuse baseline runs.

## Implementation checks

```bash
python -u scripts/check_skeleton_aux.py \
  --pretrain_ckpt /home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth
```

The 512/batch2 checks verify identical shared initialization and RNG state,
the same ImageNet tensor names, exact train/eval surface logits before training,
S2/S3 output shapes, no surface change when auxiliary predictions are changed,
the exact three-term loss, nonzero S-only encoder/decoder gradients after the
original zero readouts update, exact input-gradient ratio0.5 in each head, no
S-only gradients in the surface head, and exact strict checkpoint round trip.
The existing surface-only tests also pass.

Pure baseline parameters: 74,354,667. With two auxiliary heads: 74,985,581,
adding 630,914 parameters. The surface head and shared backbone parameter shapes
are identical. This is a correctness check, not evidence of accuracy improvement.
Detailed evidence is in `skeleton_aux_validation.json`.

## Server commands

Download the branch into its own folder:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
cd /home/gjj

curl -fL --http1.1 --retry 10 --retry-delay 3 \
  --connect-timeout 30 --max-time 900 \
  'https://codeload.github.com/ganjiajia77-maker/SwinNet/tar.gz/refs/heads/codex/e202-skeleton-aux-mass1500' \
  -o e202-skeleton-aux-mass1500.tar.gz &&
mkdir -p /home/gjj/SwinNet-e202-skeleton-aux-mass1500 &&
tar -xzf e202-skeleton-aux-mass1500.tar.gz \
  -C /home/gjj/SwinNet-e202-skeleton-aux-mass1500 --strip-components=1 &&
cd /home/gjj/SwinNet-e202-skeleton-aux-mass1500

unset RUN
export DATA=/home/gjj/Swin-Unet-main/Massachusetts/tiff
export SEED=1234
export CUDA_VISIBLE_DEVICES=1
bash scripts/run_e202_skeleton_aux_mass1500.sh train
```

After training, sweep validation thresholds without topology, then test using
the validation-selected threshold and calculate test topology:

```bash
bash scripts/run_e202_skeleton_aux_mass1500.sh sweep
bash scripts/run_e202_skeleton_aux_mass1500.sh test
```

Continue this new experiment after interruption:

```bash
bash scripts/run_e202_skeleton_aux_mass1500.sh resume
```

Find/reuse its completed topology report:

```bash
bash scripts/run_e202_skeleton_aux_mass1500.sh topology
```

Default RESULTS is `/home/gjj/SwinNet-roadbias/model_out`; PRETRAIN is the
ImageNet file shown above. DATA, RESULTS, PRETRAIN, SEED and GPU can be supplied
as environment variables. Defaults match the baseline runner, including GPU1.
The pointer is `latest_e202_skeleton_aux_mass1500_run.txt`; new experiment names
start with `e202_skeleton_aux_mass1500_`.

Unified topology uses native1500 masks, fixed Zhang–Suen, 8-neighborhood, short
area<20, APLS max64 nodes and snap radius≤5. It reports global IoU/F1/P/R and
per-image clDice, fragmentation/break metrics and both directional plus
bidirectional raster-proxy APLS, with complete image pairing checks.
