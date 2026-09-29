# P64 PSI sparse random-512 FP16 server variant

Base model/training version: `f12ff39` (P64 + PSI + real-window sparse routing with route warmup and server learning rates).

Changes in this branch:

- Adds `--amp_dtype float16` to training, validation threshold sweep, and test inference.
- Uses FP16 autocast with `torch.amp.GradScaler` during training.
- Unscales before gradient clipping and saves/restores scaler state in checkpoints.
- Keeps BF16 and FP32 modes available.
- Keeps the server learning rates: pretrained `1e-4 -> 1e-5`, new/custom `4e-4 -> 2e-5`, warmup `10` epochs.
- Keeps the model architecture, P64/PSI routing, and losses unchanged.
- Training uses one native random `512x512` crop sampled from each `1024x1024` source patch per image per epoch (`--random_crop_train --random_crops_per_image 1`).
- `threshold_sweep_test.py --fixed_crop_eval` and `test_image.py --overlap_infer` evaluate the matching native `512x512` tiles over each `1024x1024` source patch.
