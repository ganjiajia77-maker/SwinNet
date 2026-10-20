# E202 surface gap repair for native Massachusetts 1500

## Online 512-crop protocol override

This branch intentionally supersedes `docs/E202_SURFACE_GAP_REPAIR_MASS1500_SPEC.md` section 3.6 and its full-native, pre-epoch candidate refresh. The user's later instruction requires training candidates from the current random 512 crop after the decoder forward. Validation and test still use native 1500 prediction assembled from 25 overlapping 512 tiles. The online training and original full-image-refresh experiments are different protocols; their checkpoints and candidate manifests cannot be interchanged.

This branch is based directly on `eb2da24d768b7f643e4fb288e3736693862c1fe8` (`codex/e202-surface-only-mass1500`). The existing encoder, decoder, three skips, two DCA blocks, global-local bottleneck, final patch expansion and surface head are retained. No H0/S0, skeleton head, S/C/D loss, high-resolution side stream, anchor attention, or diffusion module is instantiated by the surface-only base.

The new `SurfaceGapRepair` reads two base features: stage-1 encoder output before patch merging (expected `[B,96,128,128]` at 512 input) and final decoder output before the surface head (expected `[B,96,512,512]`). Both are projected to 16 channels. RGB, tile `sigmoid(L0)`, two sigma-3 anchor maps and geometry enter a stride-2 local stem. At 128 resolution the 64-channel trunk has three residual GroupNorm blocks with dilation 1, 2 and 4. The classification head receives pooled trunk and two anchor pooled features plus a 64-D geometry embedding. The pixel head upsamples through 256 and 512 grids, uses a 1×1 58-to-16 projection and a 3×3 output convolution, and emits `delta=4*tanh(r)`; its final convolution is zero-initialized. The repair network has 299,314 trainable parameters.

Candidates come only from each current 512 tile probability at threshold 0.45, fixed Zhang-Suen skeletonization, and 8-neighbour endpoint detection. Training uses the augmented random 512 crop; validation/test apply the same candidate threshold, endpoint rule, border margin, search budget and four-candidate cap separately to each 512 sliding tile. Candidates within 8 pixels of a tile border are ignored to avoid crop-boundary false endpoints, and both anchors must have at least 16 pixels of tile context. Overlapping tiles may propose the exact same native-coordinate pair; those duplicates are assigned once to the tile with more anchor context. This avoids duplicate repair, but tile context can change near boundaries. Endpoint distances are 8–128 pixels, tangent tolerance is 75 degrees, and candidates use endpoint–endpoint and endpoint–road searches. Same locally connected paths are skipped. Prompts use pixel sigma 3. Logs report proposal counts and truncation; these are not complete gap-coverage measures. Junction-pixel cluster merging and short-spur filtering are not implemented.

Candidate search has explicit CPU budgets on each 512 tile. A predicted mask with maximum inscribed-road radius greater than 64 pixels yields no candidates for that tile, while final surface supervision still trains the base network. Otherwise at most 512 endpoint sources are searched; each source checks at most 16 ranked endpoint pairs, 1024 nearby skeleton samples and 32 road target bins. Both training and inference use these limits and keep at most four candidates per tile. Truncation may omit real gaps and is recorded.

Candidate generation does not accept GT. After generation, training labels check that anchors lie within 3 pixels of GT skeleton and that a local 8-connected GT route exists inside a bounded ROI expanded by 12–32 pixels, with route length no greater than twice the anchor distance plus 16 pixels, while the base prediction is locally gapped. A missing GT target anchor or no local route with at most 50% straight-corridor coverage is negative; ambiguous candidates are ignored. The proposal shape itself remains straight between anchors, so strongly curved connections outside the bounded ROI can be missed. Inspect `offline_gt_label_for_diagnosis_only` and candidate overlays before interpreting repair rates.

Each epoch samples one random 512 crop per native 1500 training image, batch 2, with the dataset's shared epoch value for persistent workers. The current train-mode decoder logits from each batch produce detached candidate coordinates. There is no full-image training pre-scan, extra candidate-tile forward or old numerical prediction cache. The repair network reads the same live logits, encoder/decoder features and image; candidate-weighted correction yields P1, while no candidate gives exactly P1=P0. Geometry augmentation precedes the crop and all inputs and GT remain aligned. In the initial untrained stage, no valid candidates is expected on some crops; final-surface losses still train the base network.

The first optimizer step does not wait for 25 forwards per training image. Validation/test use 25 sliding 512 forwards per native 1500 image, generate candidates on those tile outputs, then stitch original or repaired tile logits with taper0.1 weights. Repair-bearing inference tiles require another current base forward for live features. Candidate extraction is identical per 512 tile in training and inference, but random-crop context and sliding-window context differ; border exclusion may miss a gap unless another overlapping inference tile contains both anchors.

Training is joint from epoch 1 through 80, FP32, no EMA, seed 1234, AdamW weight decay 0.01, gradient clipping 1.0, warmup 10 epochs then cosine decay. Successfully loaded ImageNet tensors use LR `5e-5 -> 5e-6`; other base and repair parameters use `2e-4 -> 1e-5`. The ImageNet checkpoint hash and exact loaded tensor names are written to provenance and checkpoints. The default is the same ImageNet SwinV2 Tiny file and encoder loading rule as the baseline; no completed road checkpoint is used to initialize this run.

Loss is `L_surface(P1) + 0.1*L_soft_clDice(P1) + 1.0*L_connection` by default. Surface loss is Focal BCE gamma 1 plus 0.5 Dice on the final output for every training crop. Soft-clDice is also on every final crop, FP32 with 10 soft-skeleton iterations and smooth 1e-6, evaluated inside a 32-pixel-safe interior after full-crop skeletonization. If there are no candidates, P1=P0 and both surface losses still train the base network. Connection BCE uses only valid candidate labels with class balancing. The legacy CLI name `--lambda_repair` now controls the connection BCE weight; `--lambda_cldice` controls soft-clDice. Baseline-only mode has no repair network or connection head, but keeps final-surface and soft-clDice supervision, so it is a loss control rather than a matched-candidate-exposure control.

For multiple candidates in one tile, training uses `sum(W*p*delta)/max(1,sum(W))` with continuous classifier probabilities. Inference initially accepts candidates at `p>=0.5` and merges accepted candidates with the same weighted mean. All candidates come from the original unrepaired tile probabilities; one repair pass is applied to each tile, and the tile logits are then stitched at native 1500 resolution. Base/repaired logits, masks, candidate JSON and visualizations are saved. Candidate inference uses a second base forward only on candidate-bearing tiles for current features; it does not use stale features.

The unified metric tool is `evaluation/road_comparison/compare_predictions.py`, version `road-mask-v1.1`, with native image size 1500, 8-connectivity, short area 20, raster APLS max 64 nodes and snap radius 5. APLS is explicitly a raster approximation. Validation thresholds 0.05–0.95 at 0.05 increments are selected on global IoU; test uses only the selected threshold for the main repair result. The report also includes the base model at 0.45 and the repaired result at fixed 0.45 for an equal-threshold comparison. Test topology is calculated by the same unified tool.

## Server

```bash
cd /home/gjj
curl -fL --retry 10 --retry-delay 3 --connect-timeout 30 --max-time 900 \
  -o e202_surface_gap_repair_mass1500.tar.gz \
  https://codeload.github.com/ganjiajia77-maker/SwinNet/tar.gz/<COMMIT_SHA>
mkdir -p SwinNet-e202-surface-gap-repair-mass1500
tar -xzf e202_surface_gap_repair_mass1500.tar.gz \
  -C SwinNet-e202-surface-gap-repair-mass1500 --strip-components=1
cd /home/gjj/SwinNet-e202-surface-gap-repair-mass1500
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
python -m py_compile surface_gap_repair_experiment.py networks/surface_gap_repair.py \
  networks/vision_transformer.py networks/swin_transformer_unet_skip_expand_decoder_sys.py
```

```bash
export RUN="e202_gap_online512_v2_mass1500_fp32_joint80_s1234_$(date +%Y%m%d_%H%M%S)"
bash scripts/run_e202_surface_gap_repair_mass1500.sh train
# On interruption, keep RUN unchanged and resume from last.pth:
bash scripts/run_e202_surface_gap_repair_mass1500.sh resume
# After training:
bash scripts/run_e202_surface_gap_repair_mass1500.sh sweep
bash scripts/run_e202_surface_gap_repair_mass1500.sh test
bash scripts/run_e202_surface_gap_repair_mass1500.sh topology
```

The script defaults to physical GPU 2 via `CUDA_VISIBLE_DEVICES=2` (logical device 0 inside Python), data `/home/gjj/Swin-Unet-main/Massachusetts/tiff`, ImageNet weights `/home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth`, and output `/home/gjj/SwinNet-roadbias/model_out`. Training checks for at least 5 GiB free before starting because it stores optimizer state plus atomic `last.pth` and `best.pth` files. Set `RUN` explicitly when resuming or evaluating. To run the additional baseline-only control, choose a separate run name and invoke:

```bash
export RUN="e202_surface_gap_baseline_only_mass1500_s1234_$(date +%Y%m%d_%H%M%S)"
bash scripts/run_e202_surface_gap_repair_mass1500.sh baseline-only
```

The 0.45 candidate threshold is a configured starting value. The checked-in files did not include the historical baseline validation-selection artifact that verifies it, so the provenance must not describe 0.45 as historically verified until that record is supplied. Fixed-gap coverage, new breaks and false-connection diagnostics against an independently trained baseline A are not implemented yet. Current candidate/path counters are exploratory and must not be reported as those protocol metrics.

No long training or full test was started during development. The server ImageNet path must exist and be recorded; any optional historical surface-only reference checkpoint should be supplied separately for a direct A/B reference comparison.
