# Standard Conv-C ablation of e2020d9 on Massachusetts

The only architecture change is the Stage 2/3 connectivity prediction head:
`connectivity_context(F) -> Conv2d(96, 8, 1)`. The original eight-direction
pair feature and S/D priors are absent from C prediction. The C labels, loss,
gate, and global topology inputs remain unchanged. S/D predictions and their
supervision remain enabled.

The baseline and ablation use the same seed, Swin pretraining, native 1500
images, one random 512 crop per image per epoch, FP32, physical batch 2,
accumulation 1, 80 epochs, and no EMA. For validation and test, a 512 tile
with stride 256 is used on each full image. Threshold selection uses validation
only. The test and topology scores use the same unified road-mask tool.

```bash
cd /home/gjj/SwinNet-e202-conv-c-mass1500
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
export CUDA_VISIBLE_DEVICES=1
export SEED=1234
unset RUN  # Clear any run name exported for the pairwise-C baseline.
bash scripts/run_e202_conv_mass1500.sh train
bash scripts/run_e202_conv_mass1500.sh sweep
bash scripts/run_e202_conv_mass1500.sh test
bash scripts/run_e202_conv_mass1500.sh topology
```

The runner records its run name in
`/home/gjj/SwinNet-roadbias/model_out/latest_e202_conv_mass1500_run.txt`.
Set `RUN` explicitly when running several seeds or when resuming:

```bash
export RUN=e202_conv_c_mass1500_random512_fp32_noema80_b2_acc1_s1234_YOUR_TIMESTAMP
bash scripts/run_e202_conv_mass1500.sh resume
```

Stage 2/3 head parameters: original pairwise 481 per stage; standard
convolution 776 per stage. The common `connectivity_context` is unchanged.
For a head-only timing on the server GPU:

```bash
python scripts/benchmark_c_heads.py --device cuda --batch_size 2 --spatial_size 128
```

This timing excludes the full model and sliding-window image assembly.
