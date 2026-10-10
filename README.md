# DARENet Data1 random-512 comparison adapter

This package trains the author's `networks.lmz.DARENet` on Data1. It does not replace
the model with another network. Download the author's repository separately and pass
its directory as `--original_repo`.

The author's repository at commit `8bf0c094903f518442efe44682a59bba862d7d5c`
does not contain `networks/lmz.py`. It does contain the tracked
`networks/__pycache__/lmz.cpython-310.pyc`. The adapter loads that bytecode with
Python 3.10. A later `networks/lmz.py` source file will take priority automatically.
The model body therefore cannot currently be audited or edited as normal source.

## Data and training

- `data1/{train,val,test}/image` plus matching `mask` or `label` directories.
- Each image and GT is required to be native 1024×1024; no resizing.
- Each train image contributes exactly one 512×512 crop per epoch. Per-image crop
  locations cycle through a seeded permutation, so epochs do not repeat the same
  location until all 263169 positions have been used.
- Workers are recreated each epoch (`persistent_workers=False`) so they see the
  current epoch. Optional paired flips and 90-degree rotations follow cropping.
- Images use the original DARENet loader's BGR `pixel/255*3.2-1.6` transform.
  Targets are strict binary 0/1. Training loss is BCE-with-logits plus soft Dice.
- This adapter uses FP32, AdamW (2e-4), 5-epoch warmup, cosine decay, no EMA.
  It uses the author's model and preprocessing, with a controlled Data1 training
  schedule; it is not a claim that the original training script is identical.

## Validation, testing, and topology

Each 1024 image is covered by nine 512 windows with stride 256. Window logits are
merged with positive taper weights, divided by weight sum, then passed through
sigmoid and thresholded. Exported masks are native 1024×1024 binary PNGs.

`evaluation/road_comparison/compare_predictions.py` and `road_metrics.py` are the
shared unified tool. Its validation stage selects a threshold by global IoU from
the exported threshold grid. Its test stage checks exact case pairing and reports
surface metrics and topology including directional/bidirectional APLS with fixed
Zhang-Suen skeletonization, 8-neighborhood, area `<20`, 64 nodes, and snap radius 5.

The author's DARENet bytecode appears to instantiate an ImageNet pretrained
torchvision ResNet34. Its official `IMAGENET1K_V1` checkpoint is
`resnet34-b627a593.pth` in Torch's model cache. Existing PyTorch and torchvision
versions must match. The binary model may also require `timm`, `torchinfo`,
`ptflops`, and `mmengine` at import time.
