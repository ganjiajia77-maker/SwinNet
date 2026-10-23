# a4001b3 E2-direction encoder evidence in local-gap candidate discovery

This is a paired diagnostic, not another full-model training run. The P2
candidate generator and F2 feature belong to the `39e1398` local-gap model.
`a4001b3` does not have that P2 gap-candidate head, so the diagnostic reads
the a400 E/H/F features from its separate checkpoint. The E2 direction
aggregation is active in that checkpoint. The two backbones remain frozen.

The old `1af9580` seven-probe checkpoint must not be used: it was trained
against different encoder weights and its architecture does not contain the
E2 direction aggregator. `a400_probe_experiment.py` strictly loads the a400
checkpoint and trains ten independent 1x1 road heads on that same frozen
backbone: E1, E2, E3, H0, H_E2, H_E3, F1, F2, F3, and surface_prehead.
Each head is selected by fixed-center validation-crop IoU; test is not used
for head training or diagnostic selection.

E1 is the 128x128 encoder stage-1 output. H0 is the adapted `z_struct` before
E/H exchange, also 128x128. F1 is decoder `layers_up[1]` output after its
1/16 skip and upsampling to 64x64. There is no separate
`enable_e128_stage_fusion` branch in a400; ordinary 128x128 E1 still exists.

Validation 1 compares P2 alone against P2 augmented by a400 E2, H_E2,
H_E3, E1, H0, and F1 road-probe evidence. P2 endpoints remain first-priority
anchors. Evidence proposes additional endpoints and ranks straight candidate
paths; probabilities are not simply summed. The same per-tile candidate
budget (default 64) and existing GT labeler apply to every variant.

Validation 2 builds one coordinate-deduplicated candidate union per tile and
trains six identical-capacity MLP scorers on frozen path features: R0=F2,
R1=F2+H_E2, R2=F2+H_E2+H_E3, R3=F2+E1, R4=F2+H0, and R5=F2+F1.
All scorers see the same candidates, labels, training crops, and validation
tiles; excluded feature blocks are set to zero. No score is written back to
the surface. Candidate coverage and edge AP are diagnostic measurements,
not a measured segmentation improvement.
Tiles with no candidates keep a `(0, feature_dim)` descriptor matrix and
still contribute their verified-gap count with zero covered gaps.

## Server commands

Use the exact a400 checkout and the checkpoint from the 80-epoch run. Set
`GAP_CKPT` to the `39e1398` local-gap checkpoint already used for P2. The
separate ten-probe heads are trained once on a400 before diagnosis.

```bash
cd /home/gjj/SwinNet-h0-gap-encoder-probe-diagnostics
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet

DATA=/home/gjj/Swin-Unet-main/Massachusetts/tiff
A400_REPO=/home/gjj/SwinNet-h0-e2-direction-a4001b3
A400_CKPT=/home/gjj/SwinNet-roadbias/model_out/mass_h0_e2dir_no_decoder_d_fp32_s1234_20261022_233816/best.pth
GAP_CKPT=/path/to/39e1398-local-gap-run/best.pth
PROBE_RUN=/home/gjj/SwinNet-roadbias/model_out/a400_probe10_$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=2 OPENCV_LOG_LEVEL=ERROR \
python -u a400_probe_experiment.py train \
  --probe_repo "$A400_REPO" \
  --root_path "$DATA" --model_path "$A400_CKPT" \
  --run_dir "$PROBE_RUN" --seed 1234 \
  --max_epochs 15 --max_train_images 256 \
  --batch_size 2 --num_workers 4

OUT=/home/gjj/SwinNet-roadbias/model_out/a400_gap_encoder_evidence_$(date +%Y%m%d_%H%M%S)
CUDA_VISIBLE_DEVICES=2 OPENCV_LOG_LEVEL=ERROR \
python -u diagnose_gap_encoder_evidence.py \
  --root_path "$DATA" --gap_model_path "$GAP_CKPT" \
  --probe_repo "$A400_REPO" --probe_model_path "$A400_CKPT" \
  --probe_head_path "$PROBE_RUN/probe_best.pth" \
  --output_dir "$OUT" --device cuda:0 \
  --candidate_pool 256 --candidate_budget 64 \
  --evidence_threshold 0.45 --max_train_images 256 \
  --head_epochs 15 --seed 1234
```

The validation pass uses native 1500 images and overlapping 512 tiles at
stride 256. The head-training pass uses one deterministic random 512 crop
from up to 256 training images. Outputs include `validation1_per_tile.csv`,
`validation2_head_training.json`, and `summary.json`.

`gap_recall` counts overlapping validation tiles, so a native gap may appear
in more than one tile. Compare variants only on the same run and candidate
budget. The changed common pool also means R0/R1/R2 cannot be compared
numerically with an older P0–P4-only run. This protocol does not isolate
the effect of direction aggregation itself; that requires same-checkpoint
forward ablations with the aggregator on and off.
