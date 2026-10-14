# H0 S+D conditioning and final-surface soft-clDice on Massachusetts

Base architecture: `599a410de8d341850e1810417307e4b3bbdfc82d`.
This branch starts from the published H0 Massachusetts package `fe3b730`.
The 10 unchanged model/loss utility files in `BASE_MODEL_SHA256.json` are
byte-for-byte identical to the original H0 commit. The four changed model/loss
files implement these additions:

- The connectivity prior is [S at center, S at neighbor, D alignment to the
  connection axis]. D uses the same normalized double-angle basis as e2020d9.
- The gate gains a learned correction driven by semantic features and |D|;
  its scalar beta starts at zero, matching e2020d9.
- Global anchor tokens include C and D. Their relation bias includes both
  connectivity similarity and double-angle direction similarity.
- Global structure features are still the UPDATED high-resolution stream
  after the original bidirectional H0 Stage 2/3 interactions.
- As in e2020d9, D alignment into C, |D| into the gate, and C/D sampled for
  global context are detached. Existing direction supervision trains D.
- Final surface loss is focal BCE (gamma1) + 0.5 Dice + 0.1 soft-clDice.
  Soft-clDice uses the final 512x512 logits and aligned 512x512 surface GT,
  10 soft skeletonization iterations, and smoothing 1e-6. Stage S2/S3 logits
  are not inputs to this auxiliary loss. Existing stage losses remain.

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
OUT=/home/gjj/SwinNet-h0-mass-sd-cldice/model_out
SEED=1234
RUN="mass_h0_sd_cldice01_seed${SEED}_fp32_80e_bs2_$(date +%Y%m%d_%H%M%S)"
printf '%s\n' "$RUN" > "mass_run_seed${SEED}.txt"

python -u mass_h0.py train --gpu 2 \
  --root_path "$DATA" --output_dir "$OUT" --run_name "$RUN" \
  --pretrain_ckpt /home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth \
  --num_workers 4 --seed "$SEED"
```

After training, including in a new terminal in the same package directory:

```bash
DATA=/home/gjj/Swin-Unet-main/Massachusetts/tiff
OUT=/home/gjj/SwinNet-h0-mass-sd-cldice/model_out
SEED=1234
RUN=$(cat "mass_run_seed${SEED}.txt")
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

The focus report includes IoU/F1, clDice, break_rate, short_pred_components,
fragment_density_per_1000_px, and all three approximate APLS directions.
Here break_rate means GT skeleton pixels outside the prediction divided by
GT skeleton pixels. Short fragment counts are per-image counts with area<20;
the summary is the mean count. Fragment density uses 1000 predicted road
pixels as its denominator. Other topology fields are per-image means.

For paired multi-seed experiments, repeat this branch and full H0 with the
same seed values (e.g. 1234, 2026, 3407), the same data split and pretraining.
Each model selects its own threshold on val. Compare per-seed test results
before reporting their mean and variation. This is a combined S+D and
surface-clDice variant; separate variants are needed to isolate each change.

Outputs include `val/val_threshold_scores.csv`, `val/val_selection.json`,
`test/test_comparison.csv`, and per-image topology CSV files. Exported masks are
under `predictions/{val,test}/threshold_0.XX/surface/`. The manifest records
checkpoint identity and source hashes. Use a fresh report directory for a
different checkpoint; nonempty mask/report directories are not overwritten.
