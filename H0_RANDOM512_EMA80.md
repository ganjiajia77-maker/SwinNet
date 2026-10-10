# H0 random512 / EMA / 80 epochs

Base model: `599a410de8d341850e1810417307e4b3bbdfc82d` (Add H0 stage skeleton feature fusion).
All `networks/`, `losses/`, and model configuration files remain identical to this commit.

- Data: data1, native 1024 images, one uniformly sampled 512 crop per image per epoch.
- Crop epoch is a shared CPU tensor: persistent fork/spawn workers observe updates.
- FP32, GPU0, batch2, accumulation2 (effective batch4), augmentation enabled, seed1234.
- 80 epochs, warmup10, validation every5 epochs and the last epoch.
- AdamW LR: ImageNet-loaded 5e-5 to 5e-6; new parameters 2e-4 to 1e-5, cosine schedule.
- EMA enabled, decay0.999. Both validation and exported masks use EMA.
- H0/Stage2/Stage3/highres/global topology flags and loss parameters are explicit in the run script.
- Checkpoints: `/home/gjj/SwinNet-h0-results/$RUN/best.pth` and `last.pth`.
- Validation/test: 512 windows, stride256, tapered logit averaging, sigmoid, no TTA.
- Export native1024 binary surface masks, no morphological postprocessing.
- Unified metric scripts are copied unchanged from the SegRoadv2 comparison package
  (UNIFIED_TOOL_REVISION.txt identifies the common tool revision).
- Validation selects the threshold by global IoU; test never selects thresholds.
- Fixed Zhang-Suen, 8-neighbor components, short area<20, sampled APLS max64 nodes,
  inclusive matching radius<=5; report GT->pred, pred->GT and per-image bidirectional scores.
  These APLS scores are raster-skeleton approximations, not official full APLS.

Activate `swinunet`, then run these stages in order:

```bash
bash scripts/h0_random512_ema80.sh train
bash scripts/h0_random512_ema80.sh val
bash scripts/h0_random512_ema80.sh test
```

`test` exports masks and computes all unified pixel/topology metrics. To recompute
topology from existing masks without GPU inference:

```bash
bash scripts/h0_random512_ema80.sh topology
```

DATA, RESULTS, PRETRAIN, RUN and CUDA_VISIBLE_DEVICES can be overridden as environment
variables. The run script records the exact run name and audit directory to support new
terminals. Each validation invocation creates a fresh audit. Do not rerun `test` in the
same audit with existing masks; the exporter rejects stale output directories.

Full candidate pixel scores: `$AUDIT/val/val_threshold_scores.csv`.
Selected threshold: `$AUDIT/val/val_selection.json`.
Selected validation topology: `$AUDIT/val/val_comparison.csv`.
Test metrics: `$AUDIT/test/test_comparison.csv`, `test_report.json`, per-image CSV.

Local checks: persistent spawn worker crop regression; unchanged unified metric tests;
command-line/compile checks; 512 H0 forward/backward smoke check. These checks do not
constitute an actual data1 training run or server evaluation.
