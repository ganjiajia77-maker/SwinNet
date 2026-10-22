# Frozen H0 seven-probe diagnosis

This experiment reads the H0 encoder road/skeleton checkpoint produced by
`codex/h0-encoder-road-skeleton` at `1af958056f5953f0eb6edbffcdb9f1a9db77ba63`.
The original model is loaded strictly, kept in evaluation mode, and frozen.
No backbone parameter enters the probe optimizer.

Seven independent `1×1 Conv` road heads are trained together on a shared
forward pass. Their inputs are E2, E3, H_E2, H_E3, the post-refinement
decoder features F2 and F3, and the input to the final surface convolution.
F2 and F3 are captured from the stage 2 and stage 3 decoder structure blocks;
both are 128×128 in this model. The surface pre-head feature is 512×512.
Every head predicts the same 512×512 road mask through bilinear logit
upsampling. The training objective is focal BCE with gamma 1 plus 0.5 Dice.

The default short run samples 256 images evenly from the sorted Massachusetts
training images. It uses one random 512 crop from each selected native 1500
image per epoch, with the original geometric and color augmentation. Persistent
workers read a shared epoch value. The original H0 remains frozen. A fixed
center 512 crop of each validation image selects each head's best epoch and
controls early stopping. This center-crop selection is a cheap stabilization
check; the main reported evaluation uses native image inference.

After training, the `sweep` command performs 512 tiles at stride 256 on every
native 1500 validation image, blends logits with the original taper, then
selects a threshold for **each** probe by global IoU from 0.05 to 0.95 in
0.05 steps. The original surface output is evaluated as an eighth reference.
The `test` command uses those fixed thresholds on native test images, saves
binary masks, and reports IoU/F1/precision/recall, GT skeleton coverage, and
coverage of the original surface's missed GT skeleton pixels. Its manifest can
be read by the repository's unified road metrics tool for clDice and the
versioned approximate APLS calculation.

A positive probe result shows road labels can be read from frozen features by
this simple head. A weak probe result does not prove that the underlying
features contain no road information. Native resolution differs among probes,
and direct comparison of the original surface head with 1×1 probes is a
diagnostic comparison rather than a controlled architecture ablation.
