# Reliable Directional Dynamics v2 on Massachusetts Roads

This experiment loads the frozen local-gap H0 base and the trained dual-recovery
Q3 checkpoint strictly. The old R3 heads are loaded for source verification but
are not called. The original H stream and H3 surface fusion stay active. Stage 2
local propagation and the endpoint-based Gap module are disabled.

The new trainable modules consume H, H3, P0 and frozen Q3. The working grid is
the actual H grid (128 x 128 for 512 x 512 inputs); the final surface remains
512 x 512. There are no E128, z_struct, explicit S2/C2/D2 inputs, or GT inputs
to inference. Q3 retains its original S2/C2 conditioning.

R selects reliable **sources**. Direction relations inspect four axes at
distances 1, 2, 4, 8 and 16 working-grid pixels. Only adjacent-grid values
are transported. W includes a no-transport choice. B proposes a road probability,
A accepts the repair inside three shared convex updates producing U3. The
working-grid logit difference `logit(U3) - logit(P0_work)` is interpolated and
added to the frozen full-resolution Surface logits. There is no final A gate
or DetailHead. When U3 equals P0_work, the final Surface is exactly P0.

The A head starts with bias -1. Its effective probability has a training-only
floor of 0.15 at epoch 1; this falls linearly to zero at epoch 10. Evaluation
always uses the raw A probability. All new trainable modules start from fresh
initialization; v1 dynamic checkpoints are rejected.

Training uses `L_seg + 0.2 * L_cont`. `L_seg` is weighted focal BCE on final
logits plus half Dice; the focal background weight increases in Q3-high GT
background. `L_cont` uses connected GT skeleton pieces whose centers are not
covered within 2 pixels by frozen P0 at threshold 0.45. It computes a stable
path soft minimum with a 3 x 3 positional tolerance, normalizes each piece,
then averages pieces within each image. R, W, B,
A, Q3 and old structure heads receive no separate loss. The GT path definition
is a training proxy and does not imply guaranteed graph connectivity.

Model selection uses the validation set at a fixed 0.6 threshold during training:

```
IoU + 0.15*clDice + 0.10*GT-to-pred APLS approximation
    + 0.05*bidirectional APLS approximation
    - 0.001*short predicted components per image
    - 0.01*background bridge proxies per image
```

The final `val` command then compares thresholds 0.35 through 0.75 using the
same score; `test` reads that selection. Both P0 and P1 use the same chosen
threshold. APLS, gap closure and background bridges are raster diagnostics,
not official vector-graph metrics.

## Server commands

Run from the downloaded experiment code directory:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
export CUDA_VISIBLE_DEVICES=1
export DATA=/home/gjj/Swin-Unet-main/Massachusetts/tiff
export BASE=/home/gjj/SwinNet-roadbias/model_out/h0_local_gap_mass1500_fp32_noema80_b2_acc1_s1234_20261022_204742/best.pth
export Q3=/home/gjj/SwinNet-roadbias/model_out/h0_q3_r3_dual_recovery_40e_20261024_105737/best.pth
export RUN_DIR=/home/gjj/SwinNet-roadbias/model_out/h0_reliable_directional_v2_$(date +%Y%m%d_%H%M%S)

python -u run_reliable_directional_mass.py smoke \
  --root_path "$DATA" --output_dir "$RUN_DIR" \
  --base_checkpoint "$BASE" --q3_checkpoint "$Q3" --device cuda

python -u run_reliable_directional_mass.py train \
  --root_path "$DATA" --output_dir "$RUN_DIR" \
  --base_checkpoint "$BASE" --q3_checkpoint "$Q3" --device cuda \
  --epochs 20 --batch_size 2 --num_workers 4 --val_interval 5 \
  2>&1 | tee "$RUN_DIR.train.log"

python -u run_reliable_directional_mass.py val \
  --root_path "$DATA" --output_dir "$RUN_DIR" \
  --base_checkpoint "$BASE" --q3_checkpoint "$Q3" \
  --dynamic_checkpoint "$RUN_DIR/best.pth" --device cuda

python -u run_reliable_directional_mass.py test \
  --root_path "$DATA" --output_dir "$RUN_DIR" \
  --base_checkpoint "$BASE" --q3_checkpoint "$Q3" \
  --dynamic_checkpoint "$RUN_DIR/best.pth" --device cuda
```

To resume from `last.pth`, use mode `resume`, the same `RUN_DIR`, and
`--dynamic_checkpoint "$RUN_DIR/last.pth"` with the same training options.
The smoke report prints the real checkpoint source checks, new/frozen parameter
counts, tensor sizes, all key gradient norms, W's measured influence on U3,
and peak GPU memory and batch time. Validation saves the selected threshold,
P0/P1 paired metrics, length-binned gap proxies and eight representative repair
panels with separate W maps. `batch_metrics.jsonl` records the missed GT
segments, selected paths, path soft minimum distribution, continuity gradients
for R/W/B/A, raw and effective A in missing roads and difficult background,
and the signed probability change at missing centerlines. Validation and test
also report the added GT/background pixel ratio, reference-gap closure,
bidirectional approximate APLS, and raw A measured with no training floor.
