# e2020d9: no S/D conditioning of C

Parent model: `e2020d9a8df311f9f680cd3637466a674d76688e` (Overlap Test).
Branch: `codex/e202-no-sd-conditioning-mass1500`.

The connectivity classifier receives only `[H_i, H_j, H_j-H_i, H_j*H_i]`.
Remove the S_i/S_j/direction-alignment embedding and its classifier inputs.
The S/D heads, their losses, connectivity context, eight neighbor offsets,
structure gates, high-resolution stream, E128 injection, H2/H3 transfer,
H3 surface fusion and global topology retain the parent implementation.
This ablates direct prediction-prior conditioning of C; structural features
can still change through the original stage feedback and shared training.
Losses, data preprocessing and random crop code are identical to the parent.
The checkpoint records `connectivity_conditioning=feature_pair_only_no_sd_v1`.
Evaluation/resume reject prior-conditioned baseline checkpoints.

Protocol: native Massachusetts 1500x1500 images, one random native 512 crop
per image/epoch; shared epoch visible to persistent workers; FP32, physical
batch 2, accumulation 1, 80 epochs, GPU0, no EMA. Use the existing train/val/test
split. Dataset root defaults to `/home/gjj/Swin-Unet-main/Massachusetts/tiff`;
override DATA if your existing Massachusetts dataset is elsewhere.
Supported layouts: train/image + train/mask (or label), or train + train_labels,
and the corresponding val/test directories. No splitting or rescaling is done.

Original ImageNet-loaded LR: 5e-5 to 5e-6. New parameter LR: 2e-4 to 1e-5.
Warmup10, cosine, validation every5 and final epoch; training threshold0.2,
best.pth selected by validation F1, last.pth saved each completed finite epoch.
Use the same SwinV2 pretrained checkpoint as the parent comparison.

Val/test: native1500, 512 tiles/stride256, last edge tile at988, weighted logit
fusion with minimum 1D weight0.1, sigmoid, no TTA or postprocessing.
Sweep validation thresholds0.05:0.05:0.95 by unified global IoU without topology.
Test uses the fixed val-selected threshold, exports native1500 masks and runs
the common road-mask-v1.1 tool: Zhang-Suen, eight-neighbor components, short
area<20, APLS max64 nodes, matching radius<=5. Report both directional and
bidirectional sampled raster APLS, which is not official full SpaceNet APLS.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
cd /home/gjj/SwinNet-e202-no-sd-mass1500
export CUDA_VISIBLE_DEVICES=0
export DATA=/home/gjj/Swin-Unet-main/Massachusetts/tiff
export RESULTS=/home/gjj/SwinNet-roadbias/model_out
bash scripts/run_e202_no_sd_mass1500.sh train
bash scripts/run_e202_no_sd_mass1500.sh sweep
bash scripts/run_e202_no_sd_mass1500.sh test
```

New terminals read `RESULTS/latest_e202_no_sd_mass1500_run.txt` automatically.
Set RUN to choose another experiment. Resume with the same RUN and `resume`;
recompute topology from existing test masks with `topology`.
Audit pointer: `$RESULTS/$RUN/latest_unified_eval.txt`.
Selected threshold: `$AUDIT/val/val_selection.json`.
Final pixel/topology metrics: `$AUDIT/test/test_comparison.csv`.
Test masks: `$AUDIT/predictions/test/threshold_<selected>/surface/`.

Training checkpoint deserialization explicitly uses weights_only=False for
PyTorch2.6+ compatibility; training hyperparameters are unchanged by this fix.
Model weights and datasets are not included in the source download.

Local verification: 512 FP32 batch2 forward/backward with the original loss;
changing S/D heads cannot change C at fixed structure features; S/D losses
retain head gradients; persistent spawn workers change crops across epochs;
TIFF dataset/runner protocol checks; synthetic val/test exports and validation
threshold selection; common metric regressions. Actual Massachusetts training
and evaluation must run on the server.
