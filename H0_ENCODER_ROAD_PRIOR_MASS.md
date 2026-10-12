# H0 encoder road-prior ablation on Massachusetts

This branch extends the Massachusetts training protocol derived from `599a410`
(`Add H0 stage skeleton feature fusion`). The control is `--enable_encoder_road_prior`
(default on); the ablation uses `--no-enable_encoder_road_prior`. The off setting
passes no A1/A2 map into the first two Patch Merging feature modulations and no
A1/A2 map into encoder Stage 3 window attention. The encoder, Patch Merging
linear/MLP reduction, H0, decoder Stage 2/3 structure heads, and global topology
remain active. A1/A2 prediction heads remain instantiated in both variants.
There is no separate A3 road-prior head in `599a410`; encoder Stage 3 only
consumes the A1/A2 bias. Its Swin attention remains active in the ablation.

The paired protocol uses native 1500 Massachusetts TIFF images, one random 512
crop per training image per epoch, and 512 sliding-window inference with stride
256. It uses FP32, 80 epochs, physical batch size 2, accumulation 1 (effective
batch size 2), no EMA, and identical ImageNet checkpoint, losses, LR schedule,
augmentation, and seed for both variants. The A1/A2 auxiliary loss weight is
0.0 for both variants. Each run selects its final threshold on its own validation
predictions by global IoU. Training selects the best checkpoint using surface
IoU at threshold 0.5, matching `baselines/h0_mass/mass_h0.py`. Test metrics use
the validation-selected threshold. The sampled
raster-skeleton APLS is an approximation, not the official SpaceNet APLS.
Mass crop coordinates follow the original H0 seeded non-repeating permutation
over valid 512 crop positions. Train-time validation and exported masks both
use the original H0 0.1-minimum taper weight when stitching 512/256 tiles.

The earlier script used accumulation 2 and a training selection threshold of
0.2. Runs made with that script have effective batch 4 and must be recorded as
a different protocol. Resuming them does not remove the difference in their
earlier training. The corrected script creates a fresh output directory.

Run the requested off ablation from a local VS Code PowerShell terminal. Use
the same seed as the existing original H0 control:

```powershell
Set-Location D:\Code\Swin-Unet-main\.worktrees\h0-no-encoder-road-prior-mass1500

& .\scripts\h0_encoder_road_prior_mass.ps1 -Mode train -Variant off -Seed 1234
```

Each training command prints `RUN_DIR=...`. Keep that exact directory for the
following commands:

```powershell
$RunDir = 'PASTE_THE_PRINTED_RUN_DIR'
& .\scripts\h0_encoder_road_prior_mass.ps1 -Mode sweep -RunDir $RunDir
& .\scripts\h0_encoder_road_prior_mass.ps1 -Mode test -RunDir $RunDir
& .\scripts\h0_encoder_road_prior_mass.ps1 -Mode topology -RunDir $RunDir
```

The sweep writes `val/val_threshold_scores.csv` and `val/val_selection.json`
under the run's latest `road_prior_audit_*` directory. Test and topology write
`test/test_comparison.csv` plus per-image CSVs there. Key columns include IoU,
F1, clDice, break rate, short predicted components, fragment density per 1000
foreground pixels, and sampled APLS. Repeating the train commands with another
common seed gives a second paired run.

On the Linux server, run the no-prior ablation from this checkout. The script
uses GPU 1 by default and prints the run and output paths:

```bash
cd /home/gjj/SwinNet-h0-no-encoder-road-prior-mass
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
export CUDA_VISIBLE_DEVICES=1
export SEED=1234
export RUN=h0_mass1500_no_encoder_road_prior_fp32_noema80_b2_acc1_s1234_$(date +%Y%m%d_%H%M%S)
bash scripts/run_h0_road_prior_mass.sh train
bash scripts/run_h0_road_prior_mass.sh sweep
bash scripts/run_h0_road_prior_mass.sh test
bash scripts/run_h0_road_prior_mass.sh topology
```

`sweep` exports binary validation masks for 0.05-0.95 and selects the threshold
on validation IoU. `test` exports native 1500 test masks at that threshold.
`topology` runs the unified metrics tool and writes a summary CSV plus per-image
CSV under the run's audit folder.
