# CoANet full model on data1: 1024 source, random 512 train crop

The full CoANet (ResNet-101 + SCM + CoA) is the best architecture in Table III of the
[authors' paper](https://mftp.mmcheng.net/Papers/21TIP-CoANetRoad.pdf). The paper uses
ImageNet initialization, SGD 0.01, momentum 0.9, weight decay 0.0005, poly power 3,
and batch 16. Here a single GPU uses microbatch 4 and four accumulation steps.
The public paper does not define a best number of epochs for data1; 100 follows
the Swin comparison run.

The data1 loader uses the same train/val/test directories as Swin. Images and masks
are centered or zero padded to 1024, with the same geometric and appearance
augmentations, mask binarization and ImageNet normalization as the Swin random512
dataset. Training uses one new 512 crop per source image each epoch. The training
loader has `persistent_workers=False`; crop coordinates and worker epoch are printed
at the beginning of every epoch. Missing `--random-crop-train` is a hard error.
Validation and test use the full 1024 image, overlapping 512 tiles at stride 256,
four flip views, and the published connection branch fusion rule.

Run the following in one shell session. A backslash must be the final character
on its line.

```bash
cd /home/gjj
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
set -o pipefail

curl -fL --http1.1 --retry 10 --retry-delay 3 \
  'https://codeload.github.com/ganjiajia77-maker/SwinNet/tar.gz/refs/heads/codex/coanet-paper-best-data1-random512' \
  -o coanet-paper-best-data1-random512.tar.gz
mkdir -p /home/gjj/CoANet-paper-best-data1-random512
tar -xzf coanet-paper-best-data1-random512.tar.gz \
  -C /home/gjj/CoANet-paper-best-data1-random512 --strip-components=1

REPO=/home/gjj/CoANet-paper-best-data1-random512
DATA=/home/gjj/Swin-Unet-main/data1
RESULTS=/home/gjj/CoANet-results
cd "$REPO"
python -u check_random_crops_data1.py --data-root "$DATA" --workers 4 --seed 1234 --samples 8
```

The crop check must end in `PASS`. It prints the coordinates for the same images
at epochs 1, 2 and 3 through real worker processes.

```bash
RUN=coanet_full_resnet101_data1_random512_paper_sgd_fp32_100e_$(date +%Y%m%d_%H%M%S)
CUDA_VISIBLE_DEVICES=1 OPENCV_LOG_LEVEL=ERROR \
python -u train.py \
  --dataset data1 --data-root "$DATA" \
  --backbone resnet --out-stride 8 \
  --base-size 1024 --crop-size 512 --random-crop-train \
  --batch-size 4 --accumulation-steps 4 --workers 4 \
  --epochs 100 --eval-interval 5 \
  --lr 0.01 --lr-scheduler poly --momentum 0.9 --weight-decay 0.0005 \
  --loss-type con_ce --seed 1234 --gpu-ids 0 \
  --val-overlap-stride 256 --val-threshold 0.1 \
  --output-dir "$RESULTS" --checkname "$RUN" \
  2>&1 | tee "$REPO/${RUN}.log"
```

After training, select the experiment-specific best checkpoint. Do not use a best
file from another CoANet run.

```bash
EXP=$(find "$RESULTS/data1/$RUN" -maxdepth 1 -type d -name 'experiment_*' | sort | tail -1)
CKPT="$EXP/best.pth"
test -s "$CKPT" || { echo "No best checkpoint: $CKPT"; exit 1; }
echo "$CKPT"
```

Sweep segmentation thresholds on **val only**. The paper fusion uses fixed
connection score cutoffs 0.9 and 2.0 as in the author's public test script.

```bash
SWEEP="$RESULTS/data1/$RUN/threshold_sweep_val"
CUDA_VISIBLE_DEVICES=1 python -u threshold_sweep_data1.py \
  --root_path "$DATA" --model_path "$CKPT" --output_dir "$SWEEP" \
  --split val --source_patch_size 1024 --tile_size 512 --overlap_stride 256 \
  --prediction_mode paper_fusion \
  --thresholds 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50
THRESHOLD=$(cat "$SWEEP/best_threshold.txt")
echo "Validation threshold: $THRESHOLD"
```

Test once at the validation-selected threshold, using identical inference settings.

```bash
TEST_OUT="$RESULTS/data1/$RUN/test_paper_fusion"
CUDA_VISIBLE_DEVICES=1 python -u test_data1.py \
  --root_path "$DATA" --model_path "$CKPT" --output_dir "$TEST_OUT" \
  --split test --source_patch_size 1024 --tile_size 512 --overlap_stride 256 \
  --prediction_mode paper_fusion --threshold "$THRESHOLD"
```

For additional clDice and fragmentation measurements on the same test masks:

```bash
CUDA_VISIBLE_DEVICES=1 python -u diagnose_connectivity_data1.py \
  --root_path "$DATA" --model_path "$CKPT" \
  --output_dir "$RESULTS/data1/$RUN/topology_test" \
  --split test --source_patch_size 1024 --tile_size 512 --overlap_stride 256 \
  --prediction_mode paper_fusion --threshold "$THRESHOLD"
```
