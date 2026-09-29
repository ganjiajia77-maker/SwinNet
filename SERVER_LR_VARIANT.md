# P64 PSI sparse routing server learning-rate variant

Base commit: f12ff39 (`Restrict sparse routing to P64 PSI with warmup`)

This branch changes only the layerwise learning-rate defaults in `train_image.py`:

- `pretrained_lr=1e-4`
- `pretrained_min_lr=1e-5`
- `new_lr=4e-4`
- `new_min_lr=2e-5`
- `warmup_epochs=10`

The model architecture, P64 + PSI routing, real-window sparse execution, E128 fusion,
H3-to-surface fusion, losses, and data pipeline are unchanged.
