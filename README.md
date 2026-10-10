# WeavingUnet Data1 baseline

The model source is the user-provided original `WeavingUnet.zip`. This branch keeps its EfficientNet-V2-S encoder, SWA/DSConv, CIWM, GIEM and MSWD decoder. Compatibility fixes: CIWM pools to the actual feature size (the source assumes a 1024 input), DSConv derives its device from the input and indexes batches correctly, and the small `mmcv.ConvModule` subset used by this model is implemented with equivalent PyTorch layers to avoid `mmcv` version-specific SiLU registration.

Training: one random native-resolution 512 crop per 1024 train image each epoch, BGR normalization `image / 255 * 3.2 - 1.6`, flips/90-degree rotations, trainable ImageNet-pretrained encoder, Dice+BCE, Adam `2e-4`, cosine learning rate, micro-batch 1 with four-step gradient accumulation (effective batch 4), 80 epochs, FP32, no EMA. Best checkpoint is selected by full-resolution val IoU at a **fixed** threshold of 0.2; threshold sweeping is a separate validation-only step.

Val/test: complete 1024 images, 512 tiles with stride 256, tapered weighted **probability** fusion (the original model returns probabilities), no TTA or postprocessing. Binarized 1024 masks are passed unchanged to the shared `road-mask-v1.1` comparison tool. Val selects threshold by global IoU; test uses that frozen threshold and reports global TP/FP/FN segmentation plus per-image topology. Its bidirectional sampled raster-skeleton APLS proxy is **not** official SpaceNet APLS.

Server setup and commands (run from this repository root):

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
python -m pip install -r requirements.txt
export CUDA_VISIBLE_DEVICES=1
export DATA_ROOT=/home/gjj/Swin-Unet-main/data1
export RUN_DIR=/home/gjj/SwinNet-roadbias/model_out/weavingunet_data1_random512_$(date +%Y%m%d_%H%M%S)
# Optional if ImageNet weights are not cached or cannot be downloaded:
# export ENCODER_CKPT=/home/gjj/pretrained/efficientnet_v2_s_imagenet.pth
bash scripts/run_server.sh train
bash scripts/run_server.sh sweep
bash scripts/run_server.sh test
```

On interruption, keep the same `RUN_DIR` and run `bash scripts/run_server.sh resume`. Train must finish successfully before sweep; sweep must finish before test. Use `best.pth` for comparison and `last.pth` only to resume. The final row is in `$RUN_DIR/unified_test_metrics/test_comparison.csv`.
