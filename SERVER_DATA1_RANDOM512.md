# SegRoadv2 B2 / data1 / random512 / unified metrics

This branch is based on Yu-zhengbo/SegRoadv2 at
ab6924ecedbe4bef36713e9b3be6d5272dec1986. The model files match the user's
D:/Code/SegRoadv2-main snapshot after newline normalization. No architecture
changes are made. B2 is the default in the author's train_finetune_deep.py;
this run is not presented as a reproduction of the paper's largest/best variant.

The comparison uses native 1024 images and labels from the same data1 splits.
Each image produces one random512 training example per epoch. Flips, 90-degree
rotations, color/blur/noise augmentation and ImageNet normalization follow the
existing OARENet/CoANet data1 adapter. Epoch is fixed in each newly constructed
dataset/loader and persistent_workers=False. Fixed image index 0 and first-batch
crop positions/worker epochs are printed every epoch. There is no source resize.

Training: FP32 (TF32 disabled), no frozen backbone, MiT-B2 ImageNet backbone
initialization, AdamW lr=1e-4, min_lr=1e-6, weight_decay=0.01, original warmup/cosine
scheduler, 100 epochs, batch2, EMA0.999, validation every5 epochs. Learning rates
are explicit and are not silently rescaled by batch size. The source loss is
class-weighted CE ([1,3]) + 0.4 * (0.4*near BCE + 0.6*far BCE); no additional Dice.

Data target correction: source get_con_3 repeatedly writes channels0,1,2 and
leaves channels3..8 zero. source_fixed preserves the source's actual offsets2/4
and 9-channel row-major neighborhood including center, and writes all channels
correctly. source_legacy explicitly reproduces the original overwrite bug.
This corrected-target run must be described as such when reporting results.

Local verification passed: two DataLoader workers use the current epoch and
produce different cross-epoch crops, while the same epoch is reproducible;
MiT-B2 initialization loaded332/388 backbone tensors (56 added DCN/offset tensors
remain new); native512 CUDA forward and both connectivity/surface/DCN/offset
backward gradients are finite. A two-epoch, batch2 FP32 fixture run saved both
checkpoints and its EMA checkpoint completed val-mask export, shared threshold
selection, and test topology. The shared evaluator's11 regression tests passed.
These are runtime checks, not a completed100-epoch data1 server experiment.

Validation/test: 512 tiles, stride256, uniform overlap average, 4 flip views.
source_fusion uses road-class softmax probabilities plus the author's raw near
connectivity-logit sum>=3 or far sum>=1.5. The surface threshold is selected on
val. --prediction_mode surface and --no_tta are available for separate ablations;
the saved evaluation config prevents changing those choices between val/test.

unified_eval_data1.py exports masks and invokes the exact shared evaluator from
codex/random512-unified-eval at e3344af. It contains no alternate metric formulas.
Native1024 masks, fixed NumPy Zhang-Suen, 8-neighbor components, short area<20,
64 APLS nodes and match distance<=5 are used. Forward/reverse/bidirectional APLS
are reported as raster-skeleton approximations. Full image/GT/prediction case
sets must match; a single-model manifest checks this model versus GT. Multiple
models must be added to one manifest to assert cross-model pairing.

best.pth is selected by full-image validation IoU at fixed threshold0.5.
last.pth saves every completed epoch, including optimizer/EMA/RNG state.
Writes are atomic. The later unified val sweep selects the final test threshold.

## Download and train

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
REPO=/home/gjj/SegRoadv2-data1-random512
mkdir -p "$REPO"
curl -fL --http1.1 --retry 10 --retry-delay 3 \
  https://codeload.github.com/ganjiajia77-maker/SwinNet/tar.gz/refs/heads/codex/segroadv2-data1-random512 \
  -o /home/gjj/segroadv2-data1-random512.tar.gz
tar -xzf /home/gjj/segroadv2-data1-random512.tar.gz -C "$REPO" --strip-components=1
mkdir -p "$REPO/weights"
curl -fL --http1.1 --retry 10 --retry-delay 3 \
  https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b2_backbone_weights.pth \
  -o "$REPO/weights/segformer_b2_backbone_weights.pth"
cd "$REPO"
CUDA_VISIBLE_DEVICES=0 bash scripts/train_data1_gpu0.sh
```

The complete adapted source and unified tool are in this branch; no second
original-code download or manual folder overlay is necessary. Do not install the
source requirements.txt wholesale over an existing working CUDA environment.
Torch/torchvision must match and support torchvision.ops.deform_conv2d on CUDA;
other runtime dependencies are numpy, Pillow, cv2, scipy and einops.

## Unified validation threshold sweep and test

After training completes, in an activated swinunet environment:

```bash
REPO=/home/gjj/SegRoadv2-data1-random512
CUDA_VISIBLE_DEVICES=0 bash "$REPO/scripts/unified_val_gpu0.sh"
CUDA_VISIBLE_DEVICES=0 bash "$REPO/scripts/unified_test_gpu0.sh"
```

The val script copies a frozen checkpoint, exports all candidate masks (each image
is inferred once), lets the unified tool select maximum global IoU, and computes
topology only at the selected val threshold. The test script automatically reads
that threshold and reuses the frozen model and inference settings.

Results: $RESULTS/$RUN/unified_metrics_TIMESTAMP/val/val_threshold_scores.csv,
val/val_selection.json, val/val_comparison.csv, test/test_comparison.csv,
test/test_report.json and per-image CSVs. CPU topology runs show progress and ETA.
latest_segroadv2_run.txt is saved by the trainer; latest_unified_audit.txt is saved
by the val wrapper after successful completion. RUN/RESULTS/DATA/AUDIT environment
variables may override the wrappers' defaults.

## Recompute interrupted topology from existing masks

```bash
RESULTS=/home/gjj/SegRoadv2-results
RUN=$(cat "$RESULTS/latest_segroadv2_run.txt")
# Set AUDIT to the evaluation directory printed by the interrupted val/test run.
CUDA_VISIBLE_DEVICES=0 python -u "$REPO/unified_eval_data1.py" \
  --split test --output_dir "$AUDIT" --metrics_only \
  --metrics_output_dir "$AUDIT/test_retry_$(date +%Y%m%d_%H%M%S)"
```

For validation use --split val. To select a threshold without val topology use
--select_threshold_only on the val entry, then run test normally. Tests do not
reselect a threshold.

## Resume training

Use the same arguments as the original training command, retain its run_name,
and add --resume /home/gjj/SegRoadv2-results/RUN/last.pth. Settings that affect the
training trajectory are checked against the saved config. An interruption inside
an epoch restarts that epoch from the last completed checkpoint.
