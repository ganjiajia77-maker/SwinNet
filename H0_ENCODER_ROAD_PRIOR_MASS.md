# H0 encoder road-prior ablation on Massachusetts

This branch extends the Massachusetts training protocol derived from `599a410`
(`Add H0 stage skeleton feature fusion`). The control is `--enable_encoder_road_prior`
(default on); the ablation uses `--no-enable_encoder_road_prior`. The off setting
passes no A1/A2 map into the first two Patch Merging feature modulations and no
A1/A2 map into encoder Stage 3 window attention. The encoder, Patch Merging
linear/MLP reduction, H0, decoder Stage 2/3 structure heads, and global topology
remain active. A1/A2 prediction heads remain instantiated in both variants.

The paired protocol uses native 1500 Massachusetts TIFF images, one random 512
crop per training image per epoch, and 512 sliding-window inference with stride
256. It uses FP32, 80 epochs, physical batch size 2, accumulation 2 (effective
batch size 4), no EMA, and identical ImageNet checkpoint, losses, LR schedule,
augmentation, and seed for both variants. The A1/A2 auxiliary loss weight is
0.0 for both variants. Each run selects its final threshold on its own validation
predictions by global IoU. Test metrics use the selected threshold. The sampled
raster-skeleton APLS is an approximation, not the official SpaceNet APLS.

Run from a local VS Code PowerShell terminal. Use the same seed in both commands:

```powershell
Set-Location D:\Code\Swin-Unet-main\.worktrees\h0-no-encoder-road-prior-mass1500

& .\scripts\h0_encoder_road_prior_mass.ps1 -Mode train -Variant full -Seed 1234
& .\scripts\h0_encoder_road_prior_mass.ps1 -Mode train -Variant off -Seed 1234
```

Each training command prints `RUN_DIR=...`. Keep that exact directory for the
following commands; repeat them separately for full and off:

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
