# DINOv2-H0 unified Data1 evaluation

This run uses the same binary-mask metric implementation as the Swin/SAM/CoANet comparison. It selects the surface threshold from validation masks and computes test segmentation and topology metrics from full-resolution 1024 masks. APLS columns are sampled raster-skeleton proxies, not official SpaceNet APLS.

```bash
cd /home/gjj/SwinNet-h0-dinov2-l16-unified
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
export CUDA_VISIBLE_DEVICES=1
RUN_DIR=/home/gjj/SwinNet-roadbias/model_out/data1_h0_dinov2_l16_random512_fp32_ema80e_20261007_184505
DATA_ROOT=/home/gjj/Swin-Unet-main/data1
CKPT="$RUN_DIR/best.pth"

python -u eval_dinov2_h0.py --root_path "$DATA_ROOT" --checkpoint "$CKPT" \
  --split val --save_val_masks --output_dir "$RUN_DIR/unified_val_predictions"
python scripts/prepare_dinov2_unified_manifest.py --run_dir "$RUN_DIR" --data_root "$DATA_ROOT"
python -u evaluation/road_comparison/compare_predictions.py \
  --manifest "$RUN_DIR/unified_manifest.json" --split val \
  --output_dir "$RUN_DIR/unified_val_metrics" --progress_every 10

python scripts/prepare_dinov2_unified_manifest.py --run_dir "$RUN_DIR" \
  --data_root "$DATA_ROOT" --selection "$RUN_DIR/unified_val_metrics/val_selection.json"
TEST_MASK_DIR=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["models"][0]["predictions"]["test"][0]["path"])' "$RUN_DIR/unified_manifest.json")
if [ ! -d "$TEST_MASK_DIR" ]; then
  python -u eval_dinov2_h0.py --root_path "$DATA_ROOT" --checkpoint "$CKPT" \
    --split test --threshold_file "$RUN_DIR/unified_selected_threshold.json" \
    --output_dir "$RUN_DIR/unified_test_predictions"
fi
python -u evaluation/road_comparison/compare_predictions.py \
  --manifest "$RUN_DIR/unified_manifest.json" --split test \
  --selection "$RUN_DIR/unified_val_metrics/val_selection.json" \
  --output_dir "$RUN_DIR/unified_test_metrics" --progress_every 10
```

If the test threshold remains 0.45, the manifest helper reuses matching masks from `test_selected_threshold/surface`, provided the saved test metadata matches the checkpoint and tiling. If the selected threshold changes, it points to `unified_test_predictions/surface` and the command generates those masks. The unified tool rejects missing cases, nonbinary masks, size mismatches, and test selection without a validation record. Each output directory must be new or empty; its metric CSV files are not model training logs.
