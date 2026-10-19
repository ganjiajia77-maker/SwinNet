# Same-checkpoint direction intervention

`diagnose_direction.py` is an inference diagnostic for the original `a668711`
`semantic_directional` checkpoint. It does not train, modify the checkpoint, or
load it into a different architecture. Training and existing evaluation code
are unchanged by this addition.

For each 512 window, the same model runs twice in eval/FP32:

- `direction_on`: saved directional diffusion, normally beta=0.5.
- `direction_off`: only the direction preference in the edge logits is disabled
  (effective beta=0). S3/D3 and the semantic guide, all weights and K/eta stay the
  same. This is not a separately trained semantic model.

The diagnostic checks that X3/G3/H2/D3 are exactly equal across the two calls.
It also checks every checkpoint parameter/buffer after the whole split; the
weights and running statistics must remain unchanged. GT is only read for
offline evaluation, never passed to the model's diffusion forward.

## Server usage

Use a fresh output directory, the original already-trained directional `best.pth`,
and the Massachusetts layout used for training. No ImageNet checkpoint is needed.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
export OPENCV_LOG_LEVEL=ERROR

DATA=/home/gjj/Swin-Unet-main/Massachusetts/tiff
OUT=/home/gjj/SwinNet-h0-s2s3-diffusion/model_out
RUN=$(cat /home/gjj/SwinNet-h0-s2s3-diffusion-a668711/diffusion_run_semantic_directional_s1234.txt)
CKPT="$OUT/$RUN/best.pth"
REPORT="$OUT/$RUN/direction_diagnostic_$(date +%Y%m%d_%H%M%S)"
ls -lh "$CKPT"
df -h "$OUT"

python -u diagnose_direction.py val --gpu 2 \
  --root_path "$DATA" --model_path "$CKPT" --output_dir "$REPORT"
python -u diagnose_direction.py test --gpu 2 \
  --root_path "$DATA" --model_path "$CKPT" --output_dir "$REPORT"

cat "$REPORT/test_fixed/test_comparison.csv"
cat "$REPORT/test_own_val/test_comparison.csv"
cat "$REPORT/test_influence_summary.json"
cat "$REPORT/test_metric_deltas.json"
```

Both branches use the exact same 512/stride256 taper-weighted logit stitching
as the original evaluation, followed by sigmoid/threshold on native1500.
Val exports thresholds 0.05..0.95 in steps of 0.05, and uses the existing unified
tool's global IoU selection without computing val topology. Test exports only
the thresholds required by the following two reports:

- `test_fixed`: the primary direction-on val threshold is held fixed for both
  branches. It diagnoses the immediate effect at one operating point.
- `test_own_val`: each branch uses its own val-selected threshold. It checks
  whether calibration can explain the difference. No threshold is selected
  on test. Both reports use `compare_predictions.py` unchanged, with Zhang–Suen,
  8-neighbor components, short area<20, approximate APLS64 nodes and snap<=5.

Only binary PNGs/CSV/JSON are saved; no checkpoint copy or full-resolution float
probability archive is created. The server still needs some free output space.
Interrupted/partial exports are not silently resumed; use a fresh report root.
Val/test verify the same checkpoint SHA256, code hashes and data path.

## Reading results

`*_influence.csv` contains one row per image; `*_influence_summary.json` contains
image means. Edge and feature statistics average the same 25 windows (including
overlap); surface statistics use the stitched native1500 pixels:

- `edge_abs_change_mean_valid`: change in valid neighbor weights; invalid
  border edges are excluded from this average.
- `delta_z_abs_change_mean`: change in the latent diffusion update.
- `delta_x_abs_change_mean`: change in the projected feature correction.
- `delta_x_on_abs_mean` / `delta_x_off_abs_mean`: correction magnitudes.
- `delta_x_change_over_x_abs_mean`: correction difference relative to X3.
- `surface_probability_abs_change_mean`: change in final road probabilities.
- `mask_flip_fraction`: proportion of native pixels changing class at the
  direction-on val-selected threshold, held fixed for both branches.

`test_metric_deltas.json` reports direction_on minus direction_off. Positive
IoU/F1/clDice/APLS and negative break_rate favor the on branch. Fewer fragments
alone do not establish better topology: deleting real roads can also lower
fragment counts, so inspect recall/clDice/APLS together.

An edge change without a downstream feature/prediction change shows little
effective influence. Downstream changes without improved metrics show that
the model uses direction, but do not show a benefit. Improvements only under
the fixed threshold may largely reflect probability calibration. Improvements
under both policies support a benefit for this trained checkpoint. They do
not replace the separately trained semantic ablation or multiple seed runs.

## Verification

Run `python tests/test_direction_diagnosis.py`. The focused checks cover
zero-initialized versus active output projection, exact restoration after an
exception, unchanged state, native1500 paired exports, both threshold policies,
the unchanged unified metric pipeline, and actual H0 512 CUDA paired inference.
Synthetic fixtures test correctness, not road segmentation accuracy.
