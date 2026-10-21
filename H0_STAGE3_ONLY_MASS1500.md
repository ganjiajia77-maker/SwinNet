# H0 Stage 3-only Massachusetts Ablation

This experiment starts from `599a410de8d341850e1810417307e4b3bbdfc82d` and removes the decoder Stage 2 structure branch. The Stage 2 decoder Swin block, PatchExpand, skip connection, and DCA remain unchanged. The only decoder structure branch is Stage 3: it predicts skeleton/connectivity/direction, receives the high-resolution H0 feature, applies the existing structure residual, and feeds the existing H3-to-surface and global-topology paths. Stage 2 structure outputs and their supervision are absent; Stage 3 retains the original loss settings.

## Fixed training protocol

- Massachusetts native 1500x1500 images and labels; one deterministic random 512x512 crop per image per epoch.
- Validation and test: 512x512 sliding windows, stride 256.
- 80 epochs, physical batch 2, accumulation 1, FP32, no EMA, seed 1234.
- Stage 2 skeleton weight 0; Stage 3 skeleton weight 0.012; direction factor 0.2; connectivity factor 3.0.
- GPU 1 by default. `CUDA_VISIBLE_DEVICES=1` makes it CUDA logical device 0 within each process.
- Threshold selection uses validation predictions; test topology metrics use that selected threshold.

The Mass protocol entry point checks the architecture ID and fixed training arguments in checkpoints. `BASE_MODEL_SHA256.json` records unchanged model/loss sources from the H0 base. The two architecture files intentionally differ for this ablation.

## Server commands

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
cd ~/SwinNet-h0-stage3-only-mass1500

export DATA=/home/gjj/Swin-Unet-main/Massachusetts/tiff
export OUT=/home/gjj/SwinNet-roadbias/model_out
export PRETRAIN=/home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth
export RUN=h0_stage3_only_mass1500_random512_fp32_noema80_b2_s1234_$(date +%Y%m%d_%H%M%S)
export RUN_DIR="$OUT/$RUN"
export EVAL_DIR="$RUN_DIR/unified"

python -u mass_h0.py train \
  --root_path "$DATA" --output_dir "$OUT" --run_name "$RUN" \
  --pretrain_ckpt "$PRETRAIN" --gpu 1 --seed 1234 --num_workers 4

python -u mass_h0.py sweep \
  --root_path "$DATA" --output_dir "$EVAL_DIR" \
  --model_path "$RUN_DIR/best.pth" --gpu 1

python -u mass_h0.py test \
  --root_path "$DATA" --output_dir "$EVAL_DIR" \
  --model_path "$RUN_DIR/best.pth" --gpu 1
```

`sweep` saves validation threshold scores and the validation-selected threshold. `test` exports test masks at that threshold and writes the unified segmentation/topology reports, including per-image topology metrics. To recompute only the test metrics from existing test masks, use `mass_h0.py topology --root_path "$DATA" --output_dir "$EVAL_DIR" --model_path "$RUN_DIR/best.pth" --gpu 1 --recompute`.

To resume after interruption, use `mass_h0.py resume` with `--model_path "$RUN_DIR/last.pth"` and the same data/output/GPU arguments; it restores the checkpoint's run directory and pretraining-path metadata.
