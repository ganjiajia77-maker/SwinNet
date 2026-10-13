# e2020d9 Massachusetts run

Base: e2020d9a8df311f9f680cd3637466a674d76688e.
The model, loss implementation and random crop algorithm are unchanged.
Keep E128, z_struct, H2/H3, H3 surface fusion and global topology.
No P64, P8, PSI or sparse routing. This is the e2020d9 model, not H0.

Protocol:
- Native 1500x1500 TIFF images; one random 512x512 crop per image/epoch.
- Original full-source skeletonization, followed by the matching crop.
- Physical batch 2, accumulation 1, 80 epochs, FP32, EMA disabled.
- Physical GPU 1 by default; CUDA sees the selected GPU as logical cuda:0.
- Same SwinV2 pretrained source, module switches, loss weights and learning rates
  as the recorded e2020d9 Overlap Test experiment.
- Validation every 5 epochs (also on the final epoch), threshold 0.2 during
  training; best.pth selected by validation F1, last.pth saved each finite epoch.
- Full native-size validation/test: 512 tiles, stride 256, final edge tiles,
  taper-weighted logits (minimum 1D weight 0.1), then sigmoid.
- Final surface threshold selected on validation global IoU, grid 0.05:0.05:0.95.
- No TTA or morphology postprocessing. Topology uses the shared road-mask-v1.1
  tool at 1500x1500: 8-connectivity, zero-padded Zhang-Suen skeletons,
  short-component threshold 20 pixels, 64 APLS sample nodes, snap radius 5 pixels.
  APLS is a sampled raster approximation, not official SpaceNet APLS.

Server setup:
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
cd /home/gjj/SwinNet-e202-mass1500
export CUDA_VISIBLE_DEVICES=1
export SEED=1234
export RUN=e202_mass1500_random512_fp32_noema80_b2_acc1_s1234_$(date +%Y%m%d_%H%M%S)
bash scripts/run_e202_mass1500.sh train
```

After training, in the same shell:
```bash
bash scripts/run_e202_mass1500.sh sweep
bash scripts/run_e202_mass1500.sh test
```
Test also calculates all topology metrics. To recalculate from saved masks:
```bash
bash scripts/run_e202_mass1500.sh topology
```

To resume a completed epoch, set RUN to the interrupted run name and use:
```bash
bash scripts/run_e202_mass1500.sh resume
```
Missing last.pth means there is no epoch checkpoint to resume. The runner fails
without exiting the interactive SSH shell. When RUN is unset, evaluation/resume
read model_out/latest_e202_mass1500_run.txt; verify the selected name in output.

Dependency additions to the existing swinunet environment: yacs, timm, einops,
PyYAML, numpy, scipy, OpenCV and tqdm. Model weights/data are not in the package.
Archive contains training/evaluation runtime files and shared metrics, without
unrelated diagnostic scripts or historical model snapshots.
