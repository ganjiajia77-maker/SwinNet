# Stage 2/3 high-resolution injection probe

This is a paired inference-only diagnostic. It does not update checkpoint
weights. Each run uses one checkpoint and the first `--max_images` sorted images
from the selected split. For each image it performs native-resolution 512-tile
inference with stride 256, tapered logit fusion, and four modes: baseline,
Stage 2 off, Stage 3 off, and both off.

The script hooks `highres_structure_fusion["2"]` and `["3"]`. It measures
`||fusion_output - decoder_input||_2 / ||decoder_input||_2` and the mean
absolute residual. To disable a stage, the hook returns the unchanged decoder
input at that stage. This leaves all other checkpoint weights and the downstream
decoder intact. H0 uses its skeleton-adapted high-resolution feature here;
e2020d9 uses its original high-resolution structure feature, so the input
source is recorded separately in each report.

Example after downloading this diagnostic package into
`/home/gjj/SwinNet-stage-injection-probe`:

```bash
cd /home/gjj/SwinNet-stage-injection-probe
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
export CUDA_VISIBLE_DEVICES=1
python -u diagnostics/probe_stage_injections.py \
  --checkpoint /home/gjj/YOUR_RUN/best.pth \
  --root_path /home/gjj/Swin-Unet-main/data1 \
  --split test --threshold YOUR_VAL_SELECTED_THRESHOLD \
  --tile_size 512 --overlap_stride 256 --weight_floor 0.05 --max_images 8 \
  --output_dir /home/gjj/YOUR_RUN/stage_injection_probe
```

Read `summary.json`, `per_image.csv`, and the `*_changed_breaks.png` maps.
`new_break_pixels_vs_baseline` indicates GT skeleton pixels covered in baseline
but missed when an injection is disabled; `recovered_break_pixels_vs_baseline`
indicates the reverse. The small fixed subset is for diagnosis only, not a
replacement for full-test topology evaluation or a retrained ablation.
Use `--weight_floor 0.05` for the original H0 overlap test and `0.10` for
e2020d9, matching their respective test scripts.
The code checkout is inferred from the absolute `args.cfg` saved in each
checkpoint. If that path no longer exists, pass `--code_root` explicitly.
