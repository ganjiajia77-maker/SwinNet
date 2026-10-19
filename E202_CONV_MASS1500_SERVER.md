# Standard Conv-C ablation of e2020d9 on Massachusetts

The only architecture change is the Stage 2/3 connectivity prediction head:
`connectivity_context(F) -> Conv2d(96, 8, 1)`. The original eight-direction
pair feature and S/D priors are absent from C prediction. The C labels, loss,
gate, and global topology inputs remain unchanged. S/D predictions and their
supervision remain enabled.

Base model: `e2020d9a8df311f9f680cd3637466a674d76688e` (Overlap Test).
The native Massachusetts runner comes from `12def5b83262d11047cc8efffbb322d3b9cd11d2`.
This is the requested combined ablation: remove direct S/D conditioning AND
replace the pairwise classifier with an ordinary convolution. The shared context
block still supplies local spatial features. There is no neighbor-pair feature
concatenation, skeleton-probability input, or direction-alignment computation in
the Conv-C path. S/D can still influence the shared network through the original
gate and feedback paths; this ablation removes their direct inputs to each C head.

The baseline and ablation use the same seed, Swin pretraining, native 1500
images, one random 512 crop per image per epoch, FP32, physical batch 2,
accumulation 1, 80 epochs, and no EMA. For validation and test, a 512 tile
with stride 256 is used on each full image. Threshold selection uses validation
only. The test and topology scores use the same unified road-mask tool.

The default seed is 1234. ImageNet-loaded parameters use 5e-5 -> 5e-6;
new parameters use 2e-4 -> 1e-5, with the original 10-epoch warmup and cosine
schedule. Start from the same ImageNet pretrained file for each fresh experiment;
do not use a trained baseline best.pth as initialization. The runner's `train`
action starts fresh; `resume` is only for an interrupted Conv-C run.

```bash
cd /home/gjj/SwinNet-e202-conv-c-mass1500
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
export CUDA_VISIBLE_DEVICES=0
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

## Targeted verification

`tests/test_conv_connectivity_ablation.py` checks the resolved server command
against the fixed Massachusetts protocol, exact equality of all retained initial
tensors, and FP32 model forward/backward. It also perturbs the actual S and D
outputs inside a decoder block: C remains bitwise identical, direction matching
is never called, and C loss still backpropagates through the shared structure
features while producing no gradient for the S/D prediction heads.

Run the checks with:

```bash
python -m unittest discover -s tests -p test_conv_connectivity_ablation.py -v
```

The GPU smoke check uses a 128 input to verify model wiring. The server training
command retains the requested 512 input; this check is not an accuracy experiment.
No 80-epoch training is launched by this verification command.
