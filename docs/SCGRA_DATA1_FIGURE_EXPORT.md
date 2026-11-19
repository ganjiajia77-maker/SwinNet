# SCGRA Stage2-global figure-data export

Scope: the completed Data1/DeepGlobe encoder-H + decoder-Stage2 global experiment,
native1024,512 windows,stride256,FP32,EMA,noTTA. This is a read-only diagnostic,
not a new training variant or a threshold sweep.

The script strictly loads the saved EMA state and verifies it equals the saved
evaluation state. It reads an existing validation-IoU threshold with the same
checkpoint SHA256. If several matching validation records disagree, explicitly
pass `--selection`. The checkpoint stays on the server.

## Run

```bash
python -u scripts/export_scgra_data1.py \
  --checkpoint /absolute/run/best.pth \
  --root_path /home/gjj/Swin-Unet-main/data1 \
  --output_dir /absolute/run/scgra_val_pilot \
  --split val --max_images 20 --export_count 3 \
  --random_repeats 10 --seed 1234 --hit_tolerance_px 2 --paired_all
```

For all500 validation images, use a new output directory and `--max_images 0`.
For an explicit set of spatial examples, use `--export_cases 100773 123456` with
actual case IDs in the evaluated cohort. Without this option, spatial exports
are hash-selected with the seed, independently of GT or model effects.

`--paired_all` runs alpha=0 for every evaluated image. Without it, all images
get anchor statistics, while OFF runs only for the spatial-export subset.
Only that subset gets large spatial arrays. Set `--export_count 0` for
statistics only. The metadata records the exact evaluated/paired/exported IDs.

Re-running the identical command resumes completed images, checking checkpoint,
code, inputs and artifact hashes. Changed settings/code require a new output
directory. Input images/GT are never resized or written back to the dataset.

## Files

- `metadata.json`: checkpoint/selection/code identities, effective protocol,
  cohort, coordinate conventions, control definitions and environment.
- `anchor_statistics_per_window.csv`: actual, probability-only and random
  selection results for every512 window.
- `anchor_statistics_per_image.csv`: whole1024-image results; exact duplicate
  native anchor coordinates are removed, with both event and unique counts.
- `anchor_summary.csv`: average random repeats within image first, then average
  images. Between-image SD is not uncertainty over independent training runs.
- `writeback_effects_per_image.csv`: same-threshold ON/OFF confusion counts and
  changes on GT/background. Nonpaired images have only ON metrics.
- `summary.json`, `export_complete.json`: verification and completion status.
- `cases/<ID>/image.png`, `gt.png`: lossless RGB and binary GT.
- `cases/<ID>/prediction_on_off.npz`: float32 native1024 probabilities and signed
  ON-minus-OFF difference, computed after weighted-logit sliding-window fusion.
- `cases/<ID>/prediction_on.png`, `prediction_off.png`: shared-threshold masks.
- `cases/<ID>/change_categories.png`: numeric categories0 unchanged,1 newTP,
  2 newFP,3 lostTP,4 removedFP. This is data, not a colour-rendered figure.
- `cases/<ID>/windows/y####_x####.npz`: Stage2 P2, normalized H-score, actual
  anchor-score map, actual top-k/FPS candidates/selections and validity,
  four-head node attention and head mean, relative-position/connectivity biases,
  ungated/gated/actual-effective residual RMS maps, comparison selections,
  and native-coordinate mapping. Spatial maps are128x128 for512 input.
- `cases/<ID>/window_info.json`: window offsets and numeric-category legend.
- `partial/<ID>.json`: per-image resumable records and artifact checksums.

## Reading the traces

Every512 window has up to32 actual valid anchors by default; top-k candidates
are up to256. Controls match the *actual* valid K separately in each window.
Probability-only selection uses exactly the same top-k/FPS implementation on
P2 alone. Random controls sample without replacement from the feature grid,
using a case/window/repeat-specific deterministic seed. GT is absent from the
forward and selection-control APIs and is used only in subsequent statistics.

Coordinates use y,x order and `offset + (grid_index+0.5)*4`. Thus grid(0,0)
corresponds to nominal input pixel centre(2,2); GT lookup uses floor. Report
both exact road hits and hits within2 *original-image* pixels. Component
coverage uses strict hits in8-neighbour GT-mask components without filtering
small components, with a second version weighted by fixed Zhang-Suen skeleton
length. Window-level lengths are restricted to the window; image-level lengths
use the full1024 GT. A component hit does not mean its entire road is covered.

`node_attention` is the real post-softmax, post-bias tensor used in the forward
calculation, shaped[heads,Kmax,Kmax]. Use `selected_valid` to exclude invalid
query/key slots before plotting; attention does not establish physical road
connections. Candidate and selected ordering are preserved for matrix labels.

`effective_residual_rms` is RMS over channels of the actual `X_after-X_before`,
including the trained alpha and the receiving P2 gate. `gated_delta_rms` excludes
alpha; `ungated_delta_rms` excludes both alpha and P2. Magnitude has no sign and
does not prove useful road repair. Use the native signed probability difference
and GT-conditioned change categories for that inspection.

OFF sets only raw_alpha to0, keeps P2/anchor computation and all other branches,
and restores the exact alpha after each window. The first inferred window must
match ordinary uninstrumented surface logits bit-for-bit. Model state and
checkpoint hashes must remain unchanged before/after the full diagnostic.

The three point-selection groups are fixed-model selection diagnostics, not
three independently trained ablations. Do not infer causal benefits of an
anchor-selection design solely from these placement statistics or attention.

No publication figure is rendered by this exporter. Use the downloaded numeric
traces to create aligned panels with shared colour scales and disclose the
chosen cases, tolerance, sampled controls and ON/OFF intervention.
