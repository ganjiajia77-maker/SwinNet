# H256 encoder exchange / Stage 2 skeleton / Stage 3 diffusion

This branch starts from the verified commit `a6687113860b0bf490c5d578889b47093906ff18`.
The implementation follows [the H256 specification](docs/H0_ENCODER_H256_FROM_A668711_SPEC.md).
The new architecture ID is `h0_encoder_h256_s2_skeleton_s3_diffusion_v1`; the old
`h0_s2_skeleton_s3_directional_diffusion_v1` entry and checkpoint contract remain available.

## Architecture and initialization

At a 512 input, the shallow H stem changes its second convolution from stride 2 to
stride 1. Its output and the once-applied adapter yield 64 channels on a native
256×256 grid. Three separate E↔H modules run after encoder stages E1/E2/E3 Swin
blocks and before their PatchMerging operations. Each refines H and exchanges
features synchronously with learned coefficients initialized to 0.05. The switch
`--no-encoder_h_exchange` skips both the refinement and cross exchange.

The decoder preserves H at 256 while G2/G3 remain 128. Stage 3 supplies an
average-pooled H copy to the unchanged 128-grid directional diffusion. The original
G3→surface fusion still runs. Immediately afterward, before the original guided
surface/boundary head, one H256→16→96 residual is applied; its final projection
starts at zero. `--no-enable_h_surface_residual` skips it. The new model has no
A1/A2 heads, their PatchMerging modulation, or the E3 road bias. Normal Swin
attention, PatchMerging, encoder depth, DCA skips, bottleneck, and the original
Stage 2/3 S/D losses remain.

Fresh initialization creates the a668711 reference at the same seed and with the
same original H0 + ImageNet encoder pretraining strategy, then copies every shared
tensor with exact equality checks. The temporary reference never trains. The stem
weights are retained although stride semantics change. `initialization_manifest.json`
lists shared, removed, and new tensors. `--base_init_ckpt` is an explicit weight
migration from an old-architecture checkpoint; it never restores that optimizer.
An old checkpoint cannot resume this architecture. The checkpoint includes the
complete architecture, model config, a668711 SHA, implementation revision, and
strict model state. Training uses FP32, physical batch 2, one step per batch, 80
epochs, no EMA, one random 512 crop from each native 1500 training image per epoch,
and validation every five epochs. The random crop epoch is shared with persistent
workers. The losses are the a668711 terms: surface + 0.008 S2 + 0.012 S3 + 0.0024 D3.

## Local checks and cost

[H256_VERIFICATION.json](H256_VERIFICATION.json) contains exact initialization
tensor lists and verification evidence. Batch 1/2 forward shapes, three exchange
orders, skip and adapter identity, bidirectional formulas, final residual identity,
the FP32 batch 2 backward, gradient paths, strict checkpoint roundtrip, and
identical initialization in all three diffusion modes passed. The old diffusion
numerical checks also passed. [H256_RUNTIME_VERIFICATION.json](H256_RUNTIME_VERIFICATION.json)
records a synthetic native 1500 sweep/test/topology entry check.

On an RTX 5070 Laptop GPU with PyTorch 2.11.0+cu128, FP32 512 input:

| Model | Parameters | Batch 2 training step | Peak allocated | Peak reserved | Batch 1 inference |
| --- | ---: | ---: | ---: | ---: | ---: |
| H256 | 75,975,925 | 21.77 s | 9,637 MiB | 10,548 MiB | 191 ms |
| a668711 | 75,982,320 | 33.30 s | 8,586 MiB | 9,460 MiB | 432 ms |

Training scope: model, unchanged loss, backward, gradient clip, AdamW; one warmup
step plus three timed steps, synchronized CUDA, no I/O. Inference: two warmups and
five timed single-window forwards, synchronized CUDA, no I/O. This laptop has 8 GiB
dedicated GPU memory and uses shared memory above it; step and inference times vary
substantially between repeated measurements and cannot establish a performance
ranking. Remeasure on the server. Budget at least
about 11 GiB free GPU memory for this physical batch 2 protocol. These are compute
checks on synthetic tensors, not trained accuracy results. No 80-epoch run was started.

## Server download and run

The published branch is `codex/h0-encoder-h256-s3-diffusion`. Set `SOURCE_COMMIT`
to the full commit SHA printed by the release instruction accompanying this branch.
The source marker makes the implementation SHA available in checkpoint metadata
even from a GitHub source archive.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
set -o pipefail

REPO=/home/gjj/SwinNet-h0-encoder-h256
mkdir -p "$REPO"
curl -fL --http1.1 --retry 10 --retry-delay 3 \
  'https://codeload.github.com/ganjiajia77-maker/SwinNet/tar.gz/refs/heads/codex/h0-encoder-h256-s3-diffusion' \
  -o /home/gjj/h0-encoder-h256.tar.gz
tar -xzf /home/gjj/h0-encoder-h256.tar.gz -C "$REPO" --strip-components=1
cd "$REPO"
# Replace with the full SHA supplied with the release.
printf '%s\n' 'FULL_RELEASE_COMMIT_SHA' > SOURCE_COMMIT
```

First inspect GPU 0's free memory with `nvidia-smi`. These commands retain batch 2
and FP32. If the server GPU lacks sufficient memory, use a GPU with more memory;
changing batch or precision creates a different experimental protocol.

```bash
DATA=/home/gjj/Swin-Unet-main/Massachusetts/tiff
OUT=/home/gjj/SwinNet-h0-encoder-h256/model_out
PRETRAIN=/home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth
MODE=semantic_directional
SEED=1234
RUN="mass_h256_s2skel_s3diff_${MODE}_s${SEED}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
printf '%s\n' "$RUN" > "$OUT/latest_h256_run.txt"
cd "$REPO"

python -u h256_experiment.py inspect --gpu 0 \
  --output_dir "$OUT/${RUN}_inspection" --pretrain_ckpt "$PRETRAIN" \
  --diffusion_mode "$MODE" --seed "$SEED"

python -u h256_experiment.py train --gpu 0 \
  --root_path "$DATA" --output_dir "$OUT" --run_name "$RUN" \
  --pretrain_ckpt "$PRETRAIN" --diffusion_mode "$MODE" \
  --seed "$SEED" --num_workers 4 \
  2>&1 | tee "$OUT/${RUN}.log"
```

The training outputs are `$OUT/$RUN/best.pth`, `last.pth`, `losses.csv`,
`resolved_config.json`, and `initialization_manifest.json`. Best is selected by
validation global F1 at threshold 0.5. A separate validation sweep chooses the
threshold by global IoU. The sweep omits topology; the fixed-threshold test then
computes unified topology on native 1500 masks. Prediction uses 512 windows,
stride 256, weighted-logit stitching with floor 0.1, and no TTA or postprocessing.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
REPO=/home/gjj/SwinNet-h0-encoder-h256
DATA=/home/gjj/Swin-Unet-main/Massachusetts/tiff
OUT="$REPO/model_out"
RUN=$(cat "$OUT/latest_h256_run.txt")
CKPT="$OUT/$RUN/best.pth"
REPORT="$OUT/$RUN/unified"
cd "$REPO"

python -u h256_experiment.py sweep --gpu 0 \
  --root_path "$DATA" --model_path "$CKPT" --output_dir "$REPORT"
cat "$REPORT/val/val_threshold_scores.csv"
cat "$REPORT/val/val_selection.json"

python -u h256_experiment.py test --gpu 0 \
  --root_path "$DATA" --model_path "$CKPT" --output_dir "$REPORT"
cat "$REPORT/test/test_comparison.csv"
```

`test_comparison.csv` reports IoU/F1/P/R, clDice, components, break metrics, and
single/reciprocal/bidirectional approximate APLS via the unified evaluator
(Zhang–Suen, 8 neighbors, short component area <20, at most 64 APLS nodes,
matching distance ≤5). For a later terminal, `python -u h256_experiment.py topology
--gpu cpu --root_path "$DATA" --model_path "$CKPT" --output_dir "$REPORT"` prints the
saved test comparison. The `topology` action runs the comparison if predictions
exist but metrics are absent.

To resume the same run, use its `last.pth` and keep the mode and seed:

```bash
RUN=$(cat "$OUT/latest_h256_run.txt")
python -u h256_experiment.py train --gpu 0 \
  --root_path "$DATA" --output_dir "$OUT" --run_name "$RUN" \
  --pretrain_ckpt "$PRETRAIN" --diffusion_mode "$MODE" --seed "$SEED" \
  --num_workers 4 --resume "$OUT/$RUN/last.pth"
```

The `identity` and `semantic` diffusion controls use the same H256 stem, encoder
exchanges, A1/A2 removal, and final H residual. Run each control from fresh
initialization with the same seed. This version changes several paths at once;
overall accuracy changes alone cannot isolate a single change.
