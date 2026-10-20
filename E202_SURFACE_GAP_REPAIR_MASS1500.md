# E202 surface gap repair for native Massachusetts 1500

This branch is based directly on `eb2da24d768b7f643e4fb288e3736693862c1fe8` (`codex/e202-surface-only-mass1500`). The existing encoder, decoder, three skips, two DCA blocks, global-local bottleneck, final patch expansion and surface head are retained. No H0/S0, skeleton head, S/C/D loss, high-resolution side stream, anchor attention, or diffusion module is instantiated by the surface-only base.

The new `SurfaceGapRepair` reads two base features: stage-1 encoder output before patch merging (expected `[B,96,128,128]` at 512 input) and final decoder output before the surface head (expected `[B,96,512,512]`). Both are projected to 16 channels. RGB, tile `sigmoid(L0)`, two sigma-3 anchor maps and geometry enter a stride-2 local stem. At 128 resolution the 64-channel trunk has three residual GroupNorm blocks with dilation 1, 2 and 4. The classification head receives pooled trunk and two anchor pooled features plus a 64-D geometry embedding. The pixel head upsamples through 256 and 512 grids, uses a 1×1 58-to-16 projection and a 3×3 output convolution, and emits `delta=4*tanh(r)`; its final convolution is zero-initialized. The repair network has 299,314 trainable parameters.

Candidates come only from the current base prediction at probability 0.45, fixed Zhang-Suen skeletonization, and 8-neighbour endpoint detection. Endpoints trace inward up to 16 native pixels; those within 8 pixels of the image border do not initiate repair. Candidate endpoint distances are 8–128 native pixels, tangent tolerance 75 degrees when a tangent is reliable, maximum eight candidates per initiating endpoint and 256 per image. Endpoints and skeleton road samples produce endpoint–endpoint and endpoint–road candidates. Same locally connected paths are skipped. Candidate tiles use the original 512/256 grid, including the final edge-aligned tile; both anchors must fit with at least 16 pixels of context. Ties prefer the tile with the greatest minimum anchor-to-edge margin, then grid order. Prompts use native-pixel Gaussian sigma 3. Logs report proposal counts before and after de-duplication/truncation; they do not call these counts candidate coverage.

Candidate generation does not accept GT. After generation, training labels check that anchors lie within 3 pixels of GT skeleton and that a local 8-connected GT route exists inside a bounded ROI expanded by 12–32 pixels, with route length no greater than twice the anchor distance plus 16 pixels, while the base prediction is locally gapped. A missing GT target anchor or no local route with at most 50% straight-corridor coverage is negative; ambiguous candidates are ignored. The proposal shape itself remains straight between anchors, so strongly curved connections outside the bounded ROI can be missed. Inspect `offline_gt_label_for_diagnosis_only` and candidate overlays before interpreting repair rates.

Each epoch refreshes candidates on train images using current base weights in eval/no-grad mode. The ordinary stream still samples one random native 512 crop per 1500 image per epoch, batch 2, using the baseline dataset's shared epoch value for persistent workers. Candidate tiles are additional inputs (up to two unique tiles per optimizer step); all labeled candidates in a selected tile share one base forward and are merged before one pixel/clDice loss. The train stream uses synchronized baseline augmentations for ordinary crops. Candidate tile augmentation is off in this first implementation so coordinate prompts remain aligned; this is recorded in provenance.

Training is joint from epoch 1 through 80, FP32, no EMA, seed 1234, AdamW weight decay 0.01, gradient clipping 1.0, warmup 10 epochs then cosine decay. Successfully loaded ImageNet tensors use LR `5e-5 -> 5e-6`; other base and repair parameters use `2e-4 -> 1e-5`. The ImageNet checkpoint hash and exact loaded tensor names are written to provenance and checkpoints. The default is the same ImageNet SwinV2 Tiny file and encoder loading rule as the baseline; no completed road checkpoint is used to initialize this run.

Loss is `L_base + L_conn + lambda_repair*(L_pixel + lambda_cldice*L_soft_clDice)`. `L_base` is the ordinary random-crop Focal BCE gamma 1 plus 0.5 Dice. `L_conn` is class-balanced BCE over valid labels. Candidate tile `L_pixel` uses merged repair logits, Focal BCE gamma 1 plus 0.5 Dice over the union of candidate windows. Soft clDice uses the logits convention from `losses/cldice_loss.py`, FP32, 10 iterations, smooth `1e-6`; prediction and GT skeletons are computed on the full tile before evaluating the interior safe mask, and a 32-pixel border is excluded. Defaults are `lambda_repair=1`, `lambda_cldice=0.1`; `--lambda_cldice 0` turns off this term. Baseline-only control mode uses the same candidate tile exposure plus pixel/clDice losses but has no repair network or connection head.

For multiple candidates in a tile, training uses `sum(W*p*delta)/max(1,sum(W))` with continuous classifier probability. Inference accepts at `p>=0.5` initially and merges accepted candidates by the same weighted mean. It makes one pass: all candidates come from the unrepaired stitched probability, then corrected tile logits are stitched with the exact baseline taper0.1 outer-product weight and sigmoid. The base and repaired logits, threshold masks, candidate JSON and visualization are saved. Candidate inference currently performs a second base forward for candidate-bearing tiles to obtain live features; no stale feature cache is used.

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
export RUN="e202_surface_gap_repair_mass1500_random512_fp32_joint80_s1234_$(date +%Y%m%d_%H%M%S)"
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

No long training, full test, remote push or deployment is started by the development workflow. The server ImageNet path must exist and be recorded; any optional historical surface-only reference checkpoint should be supplied separately for a direct A/B reference comparison.
