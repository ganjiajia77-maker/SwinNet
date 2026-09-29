# P64 PSI sparse FP16 server variant

Base model/training version: `f06bc5f` (P64 + PSI + real-window sparse routing with server learning rates).

Changes in this branch:

- Adds `--amp_dtype float16` to training, validation threshold sweep, and test inference.
- Uses FP16 autocast with `torch.amp.GradScaler` during training.
- Unscales before gradient clipping and saves/restores scaler state in checkpoints.
- Keeps BF16 and FP32 modes available.
- Keeps the server learning rates: pretrained `1e-4 -> 1e-5`, new/custom `4e-4 -> 2e-5`, warmup `10` epochs.
- Does not change the model architecture, P64/PSI routing, losses, or dataset.
