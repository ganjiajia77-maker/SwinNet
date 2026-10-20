# E202 surface gap repair for native Massachusetts 1500

## Online 512-crop protocol override

This branch intentionally supersedes `docs/E202_SURFACE_GAP_REPAIR_MASS1500_SPEC.md` section 3.6 and its full-native, pre-epoch candidate refresh. The user's later instruction requires training candidates from the current random 512 crop after the decoder forward. Validation and test still use native 1500 prediction assembled from 25 overlapping 512 tiles. The online training and original full-image-refresh experiments are different protocols; their checkpoints and candidate manifests cannot be interchanged.

This branch is based directly on `eb2da24d768b7f643e4fb288e3736693862c1fe8` (`codex/e202-surface-only-mass1500`). The existing encoder, decoder, three skips, two DCA blocks, global-local bottleneck, final patch expansion and surface head are retained. No H0/S0, skeleton head, S/C/D loss, high-resolution side stream, anchor attention, or diffusion module is instantiated by the surface-only base.

The new `SurfaceGapRepair` reads two base features: stage-1 encoder output before patch merging (expected `[B,96,128,128]` at 512 input) and final decoder output before the surface head (expected `[B,96,512,512]`). Both are projected to 16 channels. RGB, tile `sigmoid(L0)`, two sigma-3 anchor maps and geometry enter a stride-2 local stem. At 128 resolution the 64-channel trunk has three residual GroupNorm blocks with dilation 1, 2 and 4. The classification head receives pooled trunk and two anchor pooled features plus a 64-D geometry embedding. The pixel head upsamples through 256 and 512 grids, uses a 1×1 58-to-16 projection and a 3×3 output convolution, and emits `delta=4*tanh(r)`; its final convolution is zero-initialized. The repair network has 299,314 trainable parameters.

Candidates come only from the current base prediction at probability 0.45, fixed Zhang-Suen skeletonization, and 8-neighbour endpoint detection. Training uses the current augmented 512 crop; validation/test use the stitched native 1500 prediction. Endpoints trace inward up to 16 pixels; those within 8 pixels of the relevant crop/image border do not initiate repair. Candidate endpoint distances are 8–128 pixels, tangent tolerance 75 degrees when a tangent is reliable, maximum eight candidates per initiating endpoint and 256 per native image. At most four candidates per 512 tile are actually passed to the repair network in both training and inference. The cap is deterministic after candidate ranking and its truncation is logged. Endpoints and skeleton road samples produce endpoint–endpoint and endpoint–road candidates. Same locally connected paths are skipped. Inference candidates are assigned to the original 512/256 grid, including the final edge-aligned tile; both anchors must fit with at least 16 pixels of context. Training candidates use the current crop as their tile. Prompts use pixel sigma 3. Logs report proposal counts before and after de-duplication/truncation; they do not call these counts candidate coverage. Junction-pixel cluster merging and short-spur filtering are not implemented in this version.

Candidate search has explicit CPU budgets to prevent an untrained dense surface mask from spending hours before the first optimizer step. A predicted mask with a maximum inscribed-road radius greater than 64 native pixels yields no candidates for that image, while ordinary base segmentation training continues. Otherwise at most 512 spatially distributed endpoint sources are searched; each source checks at most 16 ranked endpoint pairs, 1024 nearby road skeleton samples and 32 road target bins. These limits are recorded in provenance and per-image candidate statistics. They may omit true gaps, so results must report truncation and must not be presented as complete gap coverage. A checkpoint must be evaluated with the same budgeted candidate protocol used during training.

Candidate generation does not accept GT. After generation, training labels check that anchors lie within 3 pixels of GT skeleton and that a local 8-connected GT route exists inside a bounded ROI expanded by 12–32 pixels, with route length no greater than twice the anchor distance plus 16 pixels, while the base prediction is locally gapped. A missing GT target anchor or no local route with at most 50% straight-corridor coverage is negative; ambiguous candidates are ignored. The proposal shape itself remains straight between anchors, so strongly curved connections outside the bounded ROI can be missed. Inspect `offline_gt_label_for_diagnosis_only` and candidate overlays before interpreting repair rates.

Each epoch still samples one random 512 crop per native 1500 training image, batch 2, using the baseline dataset's shared epoch value for persistent workers. In each optimizer step the current train-mode decoder logits from these crops generate candidates with detached coordinates. There is no full-image pre-scan, no extra candidate-tile forward and no old numerical prediction cache. If a crop has at least one supervised candidate, all retained candidates, including ignored labels, use the same live logits and encoder/decoder features; continuous-probability delta merging produces one pixel/clDice loss per selected crop. The ordinary base loss is always computed. Geometric augmentation occurs before crop selection; candidates are generated from the augmented crop, so RGB, prediction and GT remain aligned. Before the model learns roads, candidate counts may be zero and repair parameters then receive no update on that step.

The first optimizer step no longer waits for 25 base forwards per training image. Every training step contains one base forward for its ordinary crop batch, followed by detached CPU candidate geometry and live repair computation. Validation/test still perform 25 sliding forwards per native 1500 image, plus repair-feature forwards for tiles containing candidates. Thus training and inference proposal contexts differ near crop boundaries; the 8-pixel border rule can omit training candidates that full-image inference can see. The per-tile cap and proposal truncation counts must be reported when interpreting repair coverage.

Training is joint from epoch 1 through 80, FP32, no EMA, seed 1234, AdamW weight decay 0.01, gradient clipping 1.0, warmup 10 epochs then cosine decay. Successfully loaded ImageNet tensors use LR `5e-5 -> 5e-6`; other base and repair parameters use `2e-4 -> 1e-5`. The ImageNet checkpoint hash and exact loaded tensor names are written to provenance and checkpoints. The default is the same ImageNet SwinV2 Tiny file and encoder loading rule as the baseline; no completed road checkpoint is used to initialize this run.

Loss is `L_base + L_conn + lambda_repair*(L_pixel + lambda_cldice*L_soft_clDice)`. `L_base` is the ordinary random-crop Focal BCE gamma 1 plus 0.5 Dice. `L_conn` is class-balanced BCE over valid labels. Crop `L_pixel` uses merged repair logits, Focal BCE gamma 1 plus 0.5 Dice over the union of candidate windows. Soft clDice uses the logits convention from `losses/cldice_loss.py`, FP32, 10 iterations, smooth `1e-6`; prediction and GT skeletons are computed on the full crop before evaluating the interior safe mask, and a 32-pixel border is excluded. Defaults are `lambda_repair=1`, `lambda_cldice=0.1`; `--lambda_cldice 0` turns off this term. Baseline-only mode has no repair network or connection head; it mines its own online candidates and adds masked pixel/clDice on the same crops. It is a loss control, not an exposure-matched replay control, because weights and proposals diverge during training.

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
export RUN="e202_gap_online512_mass1500_fp32_joint80_s1234_$(date +%Y%m%d_%H%M%S)"
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
