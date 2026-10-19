# H0 Stage 2 Location: Massachusetts Protocol

Run from this worktree in an existing environment with PyTorch, NumPy, OpenCV,
SciPy, timm, einops and yacs. No dependencies are downloaded by the runner.
The main model implementation owns train_image.py, test_image.py, the dataset,
and networks/h0_location_contract.py. The runner owns orchestration only.

## Source Identity

- Model reference: 599a410de8d341850e1810417307e4b3bbdfc82d.
- BASE_MODEL_SHA256.json is the immutable original model/loss digest list.
  It is provenance, not a check that the modified model remains unchanged.
- PROTOCOL_SOURCE_SHA256.json records original files from baselines/h0_mass,
  including protocol, dataset, train/test and metric adapters. These are hashes
  of the supplied baseline bytes, including their original line endings.
- Finalize IMPLEMENTATION_SHA256.json after all implementation changes. It must
  be a plain JSON object mapping worktree-relative filenames to SHA256 digests
  with CRLF normalized to LF. Verify after transfer/check-out on the training host too.
  The runner verifies this separate manifest before train, resume or evaluation.
  Supply another filename with --implementation_manifest if needed.
- The implementation manifest must cover all paths in BASE_MODEL_SHA256.json,
  h0_location.py, mass_protocol.py, train_image.py, test_image.py,
  datasets/dataset_road_skeleton.py, networks/h0_location_contract.py, config.py,
  configs/swin_tiny_patch4_window7_224_lite.yaml, both evaluation Python files,
  BASE_MODEL_SHA256.json and PROTOCOL_SOURCE_SHA256.json. Add any other imported
  implementation files. Do not include the implementation manifest itself.

## Commands

These commands are examples for a future authorized run. Implementation and
verification do not start formal training.

```bash
DATA=/home/gjj/Swin-Unet-main/Massachusetts/tiff
OUT=/home/gjj/SwinNet-h0-stage2-location/model_out
PRETRAIN=/home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth
LOCATION=encoder_e1  # run decoder separately with the same seed
RUN="h0_stage2_location_v1_${LOCATION}_mass_random512_fp32_80e_bs2_run1"

python -u h0_location.py inspect --location "$LOCATION"

python -u h0_location.py train --dry_run --location "$LOCATION" --gpu 1 \
  --root_path "$DATA" --output_dir "$OUT" --run_name "$RUN" \
  --pretrain_ckpt "$PRETRAIN" --num_workers 4 --seed 1234

python -u h0_location.py train --location "$LOCATION" --gpu 1 \
  --root_path "$DATA" --output_dir "$OUT" --run_name "$RUN" \
  --pretrain_ckpt "$PRETRAIN" --num_workers 4 --seed 1234

# Use last.pth to resume the same run and remaining original 80-epoch budget.
python -u h0_location.py inspect --location "$LOCATION" \
  --model_path "$OUT/$RUN/last.pth"
python -u h0_location.py resume --location "$LOCATION" --gpu 1 \
  --root_path "$DATA" --output_dir "$OUT" \
  --model_path "$OUT/$RUN/last.pth" --num_workers 4

CKPT="$OUT/$RUN/best.pth"
REPORT="$OUT/$RUN/unified"
python -u h0_location.py sweep --location "$LOCATION" --gpu 1 \
  --root_path "$DATA" --output_dir "$REPORT" --model_path "$CKPT"
python -u h0_location.py test --location "$LOCATION" --gpu 1 \
  --root_path "$DATA" --output_dir "$REPORT" --model_path "$CKPT"
python -u h0_location.py topology --location "$LOCATION" \
  --root_path "$DATA" --output_dir "$REPORT" --model_path "$CKPT"
python -u h0_location.py topology --location "$LOCATION" --recompute \
  --root_path "$DATA" --output_dir "$REPORT" --model_path "$CKPT"
```

Every invocation requires an explicit --location decoder or encoder_e1.
GPU defaults to 1; --gpu changes device selection without changing the protocol.
Inspect reads metadata and prints a proposed command without building the model
or starting training. Train/resume --dry_run prints the command without launching
a subprocess or writing run directories or latest-run pointers. A fresh-train
dry run accepts a planned pretraining path. Inspect and dry runs allow a missing
implementation manifest; an existing manifest is always verified.
Each location has a distinct model name and output-root latest-run pointer:
h0_stage2_location_v1_LOCATION_latest.json. A pointer records the intended run,
best.pth and last.pth paths; its presence does not prove training completed.
Pass a checkpoint explicitly for inspect/resume/evaluation.

## Fixed Training And Evaluation

Native 1500 RGB, one random 512 crop per image per epoch, 80 epochs, physical
batch 2, accumulation 1, FP32, no EMA, seed 1234 by default, four workers.
Keep the original augmentation, image normalization, shared epoch crop sequence
and sorted image list. Match image lists and seed between both locations.

ImageNet SwinV2 Tiny patch4/window8/256 encoder-only initialization, embed 96,
depths [2,2,6,2], heads [3,6,12,24], drop path 0.2. Match shared initial tensors
between both location modes; matching seeds alone does not prove equality.
AdamW weight decay 0.0001; actually loaded ImageNet tensors use 5e-5 to 5e-6,
new tensors use 2e-4 to 1e-5; ten-epoch linear warmup then the original cosine
schedule. base_lr=5e-4 is retained as metadata; the two parameter groups drive
the effective learning rates. Preserve loaded_pretrained_names on resume.

Training validation every five epochs and the final epoch uses global surface
F1 at threshold 0.5 to select best.pth. Afterward, the unchanged unified tool
selects a validation-only global-IoU threshold from 0.05 through 0.95 by 0.05;
ties choose the lower threshold. Test never selects a threshold.

Surface loss is the original focal BCE (gamma 1) plus 0.5 Dice. Preserve
stage=2 and stage=3 slots with weights 0.008 and 0.012; connectivity factor 3,
direction factor 0.2, skeleton pos_weight 2, connectivity pos_weight 2,
connectivity focal gamma 1.5, masked-center connectivity, reciprocal symmetry
0.05, edge contrastive margin 0.1, cardinal/diagonal positive factors 1/2.5.
Both structure-input gradient ratios are 0.5; Stage 3 gate topology ratio is 0.
High-resolution skeleton, road-attention and final auxiliary weights stay 0.
Both stage loss multipliers stay 1. Do not change the original internal losses.

Inference uses exactly 25 windows (starts 0,256,512,768,988 on each axis),
512 tiles, stride 256, taper-weighted logits, divisor clamp 1e-8, then sigmoid.
No TTA or component removal. Unified metrics use native 1500, short-component
area <20, approximate APLS max nodes 64 and snap radius 5. This raster proxy is
not the official SpaceNet APLS evaluator.

## Checkpoint And Cache Contract

The runner requires model_state_dict, epoch in 1..80, resolved args matching
mass_protocol.validate_training_args and the full reference loss/LR budget,
and architecture_contract with architecture_id=h0_stage2_location_v1,
stage2_structure_location matching --location, and base_model_commit matching
the full reference SHA. Saved args must also record architecture_id/location.
It deliberately does not use experiment_base_commit as proof of model identity.
New checkpoints may set that old field to null so the original H0 runner rejects
them. model_base_commit can retain the full reference SHA as provenance.

The main model loader must compare the entire contract, including protocol
version/source/implementation digests, model config/constructor, H channels/grid,
A1/A2 and global topology status and the two structure-unit execution counts,
before loading weights with strict=True. Same tensor names/shapes do not prove
the call location. Old H0 weights are not resumable location checkpoints.
Resume additionally requires optimizer_state_dict and loaded_pretrained_names;
the loader must reject failed optimizer restoration rather than restart it.

Sweep freezes checkpoint bytes, full args/contract, source digests, data root
and model name into comparison.json. Test/topology validate the saved selection
against that identity. Keep comparison.json and val/val_selection.json after
cleanup; val mask images are not required afterward. test computes both test
segmentation and topology. topology reuses a matching complete report without
reading masks, or --recompute writes a fresh timestamped metric directory from
existing selected test masks. Recompute therefore requires test masks.
Use a fresh report directory when checkpoint, code, location or data root changes.
The unchanged evaluator refuses nonempty metric output directories.
