# Original Swin-Unet baseline on Data1

This package keeps the supplied Swin-Unet encoder, decoder, and two-class
segmentation head. `train_data1.py`, `infer_data1.py`, and `data1_common.py`
adapt the input and training loop to Data1. The original Synapse entry points
remain available separately.

## Experiment protocol

- Source: native 1024×1024 RGB image and binary road GT.
- Train: one 512×512 crop per source image and epoch. A seeded permutation
  guarantees a different crop coordinate for each image over the first
  263169 epochs. DataLoader workers are **not** persistent, so every new
  epoch receives its new dataset epoch. Random flips and quarter turns match
  the original augmentation style.
- Model: original Swin-Unet two-class head and original 0.4 CE + 0.6 Dice loss.
  The 512 input requires window size 8: the token grids are 128, 64, 32, 16,
  all divisible by 8. Swin-T window-7 pretrained relative-position tables
  are bicubic-interpolated to window 8; matching encoder weights are also
  copied to the corresponding decoder stages, as in the original loader.
- Optimizer: SGD, momentum 0.9, weight decay 1e-4, original polynomial
  learning-rate decay; configurable base LR. Gradient accumulation enables
  an effective batch larger than one when GPU memory is limited.
- Validation/test: 512×512 sliding windows with stride 256 on native 1024
  images; taper-weighted road logit stitching, then sigmoid. No TTA or
  morphological postprocessing. The two-class road logit is class-1 minus
  class-0, exactly matching softmax road probability.
- Checkpoint selection: validation global IoU at a fixed 0.5 threshold.
  The independent validation threshold sweep happens afterward.
- FP32, no EMA.

`infer_data1.py` exports complete native-resolution binary masks for every
requested threshold. `evaluation/road_comparison/compare_predictions.py` is
the unchanged unified evaluation tool: it selects among saved validation
masks by global IoU, then reports native-resolution segmentation, connectivity,
clDice and three sampled APLS directions on the selected validation and test
mask sets. The included manifest builder only adapts this baseline's output
directories to the shared manifest format.

The pretrained checkpoint must be **Swin-T V1**
`swin_tiny_patch4_window7_224.pth`. SwinV2 window-8 weights are a different
architecture and should not be substituted. The Swin-T V1 release is linked
from Microsoft's Swin Transformer model hub. No pretrained file or Data1
dataset is stored in this branch.
