# Original H0 on Massachusetts Roads

Base architecture: `599a410de8d341850e1810417307e4b3bbdfc82d`.
The 14 model, loss, and direction-target files in `BASE_MODEL_SHA256.json` are
byte-for-byte identical to that commit. This package changes data handling,
training/evaluation entry points, and the unified metric adapter.

## Fixed experiment

- Native RGB images and binary labels: 1500 x 1500, no resizing.
- Training: one 512 x 512 crop per image per epoch, 80 epochs, batch size 2,
  accumulation 1, FP32, no EMA, seed 1234.
- Workers share the current epoch; crop coordinates change between epochs
  even with persistent workers. Each image follows a seeded permutation of
  valid crop coordinates, avoiding repeats within the 80 epochs.
- Validation/test: 512 windows with stride 256; starts 0, 256, 512, 768, 988
  on both axes, covering the whole native image with 25 windows.
- The same taper-weighted logit stitching is used for training validation,
  threshold sweeping, and test inference. Sigmoid and thresholding occur
  after stitching. No TTA or automatic component removal.
- Training validation every 5 epochs; step-based trend validation disabled.
- ImageNet SwinV2 Tiny window8/256 pretraining; no data1 checkpoint warm start.
- Original H0 learning rates: pretrained 5e-5 -> 5e-6, new 2e-4 -> 1e-5,
  warmup 10 epochs. Original H0 skeleton/connectivity/direction/surface losses.

Required dataset layout:

```
/home/gjj/Swin-Unet-main/Massachusetts/tiff/
  train/         train_labels/
  val/           val_labels/
  test/          test_labels/
```

Image and label stems must pair completely, including `.tiff` images and `.tif`
labels. Missing/extra labels and invalid image dimensions fail explicitly.

## Server commands (run from this package directory)

Use the existing `swinunet` environment. If dependencies are missing, install
`requirements-mass.txt` there; do not replace its CUDA-enabled PyTorch.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet

DATA=/home/gjj/Swin-Unet-main/Massachusetts/tiff
OUT=/home/gjj/SwinNet-h0-mass/model_out
RUN="mass_h0_599a410_random512_fp32_80e_bs2_$(date +%Y%m%d_%H%M%S)"
printf '%s\n' "$RUN" > mass_run_name.txt

python -u mass_h0.py train --gpu 2 \
  --root_path "$DATA" --output_dir "$OUT" --run_name "$RUN" \
  --pretrain_ckpt /home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth \
  --num_workers 4
```

After training, including in a new terminal in the same package directory:

```bash
DATA=/home/gjj/Swin-Unet-main/Massachusetts/tiff
OUT=/home/gjj/SwinNet-h0-mass/model_out
RUN=$(cat mass_run_name.txt)
CKPT="$OUT/$RUN/best.pth"
REPORT="$OUT/$RUN/unified"

python -u mass_h0.py sweep --gpu 2 \
  --root_path "$DATA" --model_path "$CKPT" --output_dir "$REPORT"

python -u mass_h0.py test --gpu 2 \
  --root_path "$DATA" --model_path "$CKPT" --output_dir "$REPORT"

python -u mass_h0.py topology --gpu 2 \
  --root_path "$DATA" --model_path "$CKPT" --output_dir "$REPORT"
```

## Unified evaluation

`mass_h0.py` calls `test_image.py` to export native 1500 binary surface masks,
then uses `evaluation/road_comparison/compare_predictions.py` for selection and
reporting. The unified metric implementation is unchanged; its configured
image size is 1500 for Massachusetts.

- Sweep: validation only, thresholds 0.05 through 0.95 in steps of 0.05.
  Select by global pixel IoU. No validation topology computation.
- Test: read `val/val_selection.json`, export test masks at that fixed threshold,
  report segmentation and topology with the unified tool.
- Topology: display the report already computed by `test`. Add `--recompute`
  to generate a fresh topology report from the existing masks, without GPU
  inference or overwriting the previous report.
- Fixed Zhang-Suen skeletonization, 8-neighbor connectivity, short-component
  area <20 pixels, APLS proxy maximum 64 nodes, matching distance <=5 pixels.
  Reports GT-to-prediction, prediction-to-GT, and bidirectional approximate
  APLS. This raster graph proxy is not the official SpaceNet APLS evaluator.
- Strict image/GT/prediction pairing; native resolution required.

Outputs include `val/val_threshold_scores.csv`, `val/val_selection.json`,
`test/test_comparison.csv`, and per-image topology CSV files. Exported masks are
under `predictions/{val,test}/threshold_0.XX/surface/`. The manifest records
checkpoint identity and source hashes. Use a fresh report directory for a
different checkpoint; nonempty mask/report directories are not overwritten.
