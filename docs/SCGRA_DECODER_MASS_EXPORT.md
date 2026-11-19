# Original decoder-control SCGRA mechanism export

This diagnostic uses exactly the original architecture from commit
`892e5d017c19bbf017ae28e7bccef03423f7329f`, location `decoder`, architecture ID
`h0_stage2_location_v1`. Model name:
`h0_stage2_location_v1_decoder_mass_random512_fp32_80e_bs2`.

It does not use the newer encoder-H + Stage2-global experiment, its no-GAP
migration, or the Data1 EMA checkpoint. No training is performed. All original
architecture files and model calculations are preserved; only optional detached
capture is added to `keypoint_global_topology.py`.

## Inputs and protocol

- Massachusetts native1500 RGB and GT, `val` has14 images, `test`49.
- 512 windows, stride256, starts `[0,256,512,768,988]` on each axis:25 windows.
- Original triangular weighted logit stitching (each ramp minimum0.1), FP32,
  no TTA and no EMA; strict `model_state_dict` loading and architecture contract.
- The decoder-control global module runs in `up_x4`, **after final expansion**,
  before the original H3/surface fusion. Its anchor grid is512x512, not128x128.
- The exported `road_prior` is the original surface preview at this location;
  it is not a newly added Stage2 prediction head.
- Coordinates y,x: native `offset + (grid_index+0.5)*tile_size/grid_size`.
  This version maps to pixel centres `offset+k+0.5`; floor is used for GT lookup.

The existing validation selection must match the checkpoint SHA256. The
original unified `val_selection.json` plus its adjacent `comparison.json` are
supported, including model identity verification. Deleted validation mask
directories need not be restored. Conflicting matching thresholds require an
explicit `--selection`; test predictions never select the threshold.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/export_scgra_decoder_mass.py \
  --checkpoint /actual/decoder-control/best.pth \
  --root_path /home/gjj/Swin-Unet-main/Massachusetts/tiff \
  --output_dir /actual/run/scgra_decoder_mass_val \
  --device cuda:0 --split val --max_images 0 --export_count 3 \
  --random_repeats 10 --seed 1234 --hit_tolerance_px 2 --paired_all
```

This runs all14 validation images. Spatial traces are saved for3 hash-selected
images, independently of predictions/GT, and all14 receive anchor metrics and
normal-vs-alpha-zero predictions. `--max_images 1 --export_count 1` is a pilot.
`--export_cases ID ...` can specify declared spatial examples in the cohort.
Re-run the identical command and output path to resume complete image records.
If the checkpoint directory is unknown, replace `--checkpoint FILE` with
`--checkpoint_root /home/gjj`. This reads the original decoder experiment's
latest-run pointers and identity-named run directories, skips datasets and
environments, and requires exactly one existing best.pth; multiple candidates
are printed for explicit selection. Full architecture/weight validation remains
mandatory. The output path must still be specified independently.

## Outputs

- `metadata.json`: exact input/threshold/code identities, architecture, coordinate
  convention and selected cohorts. `summary.json` records trained alpha and
  unchanged-state verification. `export_complete.json` marks success.
- `anchor_summary.csv` and `anchor_statistics_per_image.csv`: actual structure
  score times road prior + FPS, prior-only using the same FPS, and random
  same-K selection (10 deterministic repeats). Each window matches the actual
  number of valid anchors, at most32. Image statistics deduplicate exact native
  positions; random repeats are averaged within each image first.
- Road hit rates exact and within2 original pixels, nearest-neighbour distances,
  convex-hull area fraction, GT road-component coverage and coverage weighted
  by Zhang-Suen skeleton length. Components use8-neighbour GT-mask connectivity.
  Hitting a component does not mean covering its whole centreline.
- `anchor_statistics_per_window.csv`: corresponding window-level statistics.
- `writeback_effects_per_image.csv`: same-threshold ON/OFF confusion counts,
  recovered/lost TP and added/removed FP; signed GT/background probability change.
- `cases/ID/image.png`, `gt.png`, `prediction_on.png`, `prediction_off.png`.
- `cases/ID/prediction_on_off.npz`: native1500 float32 probabilities and signed
  ON-minus-OFF difference after stitching.
- `cases/ID/change_categories.png`: numeric IDs0 unchanged,1 newTP,2 newFP,
  3 lostTP,4 removedFP. These are data, not rendered colour figures.
- `cases/ID/windows/y####_x####.npz`: road prior, normalized structure score,
  true candidate and selected anchors with validity, true per-head node attention
  and head mean, biases, ungated/gated/actual-effective residual RMS maps,
  comparison selections, native-coordinate mapping. Use validity masks in plots.

Effective residual RMS is computed from actual `output-feature`, including alpha
and the receiving road-prior gate. ON/OFF changes only raw_alpha to zero and
restores it exactly per window. First-window captured and ordinary predictions
must match bit-for-bit; model tensors and checkpoint hashes must stay unchanged.

Attention is not a road edge prediction; residual magnitude does not establish
useful repair. Fixed-model point-placement controls are not retrained ablations.
Use GT-conditioned ON/OFF changes to assess effects, and disclose the selected
examples and intervention. This export does not render the publication figure.
