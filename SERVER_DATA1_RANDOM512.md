# OARENet original architecture on data1

Base: WanderRainy/OARENet, commit f203d6e. The SwinT_OAM forward architecture,
decoder, and occlusion module remain unchanged. The pretrained checkpoint path
is configurable; MMCV checkpoint/registry utilities were replaced by small
equivalent loaders so the code runs in the existing `swinunet` environment.

Data: the same `data1/train`, `data1/val`, `data1/test` split used for Swin and
CoANet. Train images and labels are centered/padded to 1024, jointly flipped
and rotated, then one randomly positioned 512 crop is drawn for each image on
each epoch. Color perturbation and ImageNet normalization match the CoANet
data1 comparison. A separate DataLoader is made each epoch and
`persistent_workers=False` so workers see the current epoch. The same image
index 0 and the first batch's crop coordinates are printed as `CROP-AUDIT`
each epoch.

Validation/test: full 1024 images, 512 sliding tiles with stride 256, mean
fusion in overlaps, four flip views per tile. Model selection uses val IoU at
fixed threshold 0.5 every five epochs. The later val sweep selects the test
threshold; the test set is evaluated once at this chosen value. Training uses
FP32, Adam 2e-4, the original factor-of-five LR drops after epochs 50, 65,
and 80, and the same optional EMA 0.999 used in the comparison experiments.

The original 22K Swin-T backbone checkpoint can be obtained from the
[official Swin model hub](https://github.com/microsoft/Swin-Transformer/blob/main/MODELHUB.md).
Swin V2 checkpoints are a different model and should not be substituted.

## Download on the server

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
REPO=/home/gjj/OARENet-data1-random512
DATA=/home/gjj/Swin-Unet-main/data1
RESULTS=/home/gjj/OARENet-results
mkdir -p "$REPO" "$RESULTS"

curl -fL --http1.1 --retry 10 --retry-delay 3 \
  'https://codeload.github.com/ganjiajia77-maker/SwinNet/tar.gz/refs/heads/codex/oarenet-data1-random512' \
  -o /home/gjj/oarenet-data1-random512.tar.gz
tar -xzf /home/gjj/oarenet-data1-random512.tar.gz -C "$REPO" --strip-components=1

mkdir -p "$REPO/weights"
curl -fL --http1.1 --retry 10 --retry-delay 3 \
  'https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth' \
  -o "$REPO/weights/swin_tiny_patch4_window7_224_22k.pth"
ls -lh "$REPO/weights/swin_tiny_patch4_window7_224_22k.pth"
```

## Train on GPU 0

```bash
cd "$REPO"
set -o pipefail
RUN=oarenet_original_data1_random512_ema0999_fp32_100e_$(date +%Y%m%d_%H%M%S)
printf '%s\n' "$RUN" > "$RESULTS/latest_oarenet_run.txt"
CUDA_VISIBLE_DEVICES=0 OPENCV_LOG_LEVEL=ERROR \
python -u train_data1.py \
  --root_path "$DATA" --output_dir "$RESULTS" --run_name "$RUN" \
  --pretrain_ckpt "$REPO/weights/swin_tiny_patch4_window7_224_22k.pth" \
  --source_patch_size 1024 --img_size 512 --overlap_stride 256 \
  --batch_size 2 --num_workers 4 --max_epochs 100 --val_interval 5 \
  --lr 2e-4 --seed 1234 --use_ema --ema_decay 0.999 \
  2>&1 | tee "$RESULTS/${RUN}.log"
```

## Sweep on val only

```bash
REPO=/home/gjj/OARENet-data1-random512
DATA=/home/gjj/Swin-Unet-main/data1
RESULTS=/home/gjj/OARENet-results
RUN=$(cat "$RESULTS/latest_oarenet_run.txt")
CKPT="$RESULTS/$RUN/best.pth"
SWEEP="$RESULTS/$RUN/threshold_sweep_val"
cd "$REPO"
CUDA_VISIBLE_DEVICES=0 python -u threshold_sweep_data1.py \
  --root_path "$DATA" --model_path "$CKPT" --output_dir "$SWEEP" \
  --split val --source_patch_size 1024 --tile_size 512 --overlap_stride 256 \
  --thresholds 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50
cat "$SWEEP/best_threshold.txt"
cat "$SWEEP/threshold_sweep_val.csv"
```

## Test at the selected threshold

```bash
REPO=/home/gjj/OARENet-data1-random512
DATA=/home/gjj/Swin-Unet-main/data1
RESULTS=/home/gjj/OARENet-results
RUN=$(cat "$RESULTS/latest_oarenet_run.txt")
CKPT="$RESULTS/$RUN/best.pth"
THRESHOLD=$(cat "$RESULTS/$RUN/threshold_sweep_val/best_threshold.txt")
TEST_OUT="$RESULTS/$RUN/test_sliding512"
cd "$REPO"
CUDA_VISIBLE_DEVICES=0 python -u test_data1.py \
  --root_path "$DATA" --model_path "$CKPT" --output_dir "$TEST_OUT" \
  --split test --source_patch_size 1024 --tile_size 512 --overlap_stride 256 \
  --threshold "$THRESHOLD"
cat "$TEST_OUT/test_metrics.csv"
```
