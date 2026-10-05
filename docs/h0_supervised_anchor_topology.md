# H0 supervised anchor topology

Base: `599a410` (`Add H0 stage skeleton feature fusion`). The H0/H1/H2
feature exchange, Stage 2/3 S/C/D heads, decoder gates, H3 surface fusion,
encoder and decoder depths are unchanged. No E128, R64, P64/PSI routing,
endpoint detector or direction-to-connectivity experiment is introduced.

Enable with `--enable_global_topology --global_topology_mode supervised_anchors`.
The legacy `feature_anchors` implementation remains available for old models,
but is not instantiated in the new mode.

## Predicted relationships

- The original H2 feature norm times predicted surface probability selects up
  to 32 FPS anchors from 256 scored candidates. This does not detect endpoints.
- Each anchor nominates at most four neighbours within 64 pixels, at most 128
  directed nominations per image. Undirected canonicalization ensures identical
  logits for both directions. Candidate attention is symmetric and keeps self
  edges; incoming nominations can make a node's symmetric degree exceed four.
- Decoder/H2 features, C3 probabilities, S3 probabilities, surface probabilities
  and relative geometry form the evidence. Each pair samples eight ordered
  positions on three lanes (offsets -2, 0, +2 pixels). A 1D convolution retains
  order, then an MLP independently predicts sigmoid local-reachability scores.
- Node attention uses relative-position bias and a ramped log-probability prior.
  Only nominated pairs and self edges participate.
- Learned node increments and path gates are bilinearly scattered only into
  the narrow corridors, including the anchor positions. Overlaps are normalized
  without canceling edge confidence. The full original feature is the identity
  path; there is no hard road filling and no full-grid cross-attention to nodes.

## Supervision and cache

GT skeletons are derived from the resized 256 road mask **for the new edge
task only**. The pre-existing Stage 2/3 skeleton target pipeline is unchanged.
Skeleton pixels form an eight-neighbour graph with edge lengths 1 or sqrt(2).
Bounded geodesics (96 pixels) are computed once in memory-bounded chunks and
stored as sparse sorted distance lookups. Empty masks are valid cache entries.

Predicted anchors snap within three pixels. Near-ties between distant skeleton
branches are ignored. Pairs with unmatched/ambiguous/duplicate snap positions
are ignored, not treated as background negatives. A valid pair is positive only
if geodesic length <=96 and <=2*Euclidean distance+2 pixels. All other valid
pairs are negative, including disconnected roads and excessive detours within
one connected component. This is local reachability, not a vector road edge.

The cache hashes the actual post-geometry resized binary mask, resolution and
geodesic settings, so flips/rotations/crops cannot reuse misaligned targets.
Only a vectorized lookup of nominated pairs runs in the training step. No GT
is used in model forward, candidate selection, attention or inference.

`prepare_anchor_reachability.py` optionally prepares all eight train geometry
variants before training with progress output. The one-off CPU preparation can
be slow and consumes disk space; it is not part of epoch/inference timing.
Training also creates missing cache entries lazily. SciPy is required for cache
construction, but not for network inference.

Total training loss = original H0 loss + 0.1*balanced connection BCE. Positive
and negative class means are averaged when present. Empty/one-class batches
are finite; counts are recorded so missing negative coverage is visible.
Existing surface/skeleton/connectivity/direction supervision remains full-image.

First 10 epochs train the connection head without new residual writeback.
Epochs 11-15 ramp connection prior/writeback from 0.2 to 1.0. This warmup is
independent of the unchanged LR warmup. The residual scale is zero-initialized
and bounded by the original alpha_max=0.05. Learned gates and confidence
control writes; GT paths are never drawn onto predictions.

## Running

Activate the `swinunet` environment and set `RUN`. The server wrapper defaults
to GPU 1, FP32, 1024->256 direct resize, batch 4, 100 epochs, no EMA, and the
existing H0 loss weights/LR groups. `MAX_EPOCHS`, `CUDA_VISIBLE_DEVICES`,
`WORKERS`, `DATA_ROOT`, `OUTPUT_ROOT` and `PRETRAIN` can override server paths.

```
bash scripts/run_h0_anchor_topology_server.sh cache
export RUN=data1_h0_supervised_anchors_fp32_100e_$(date +%Y%m%d_%H%M%S)
bash scripts/run_h0_anchor_topology_server.sh train
# Only after training:
bash scripts/run_h0_anchor_topology_server.sh sweep
bash scripts/run_h0_anchor_topology_server.sh test
# After an interruption, with the same RUN/MAX_EPOCHS:
bash scripts/run_h0_anchor_topology_server.sh resume
```

Sweep uses **val**, accumulating global pixel TP/FP/FN, like test. It stores
`threshold_val.json`. Test reads the selected surface threshold automatically
unless `BEST_THRESHOLD` is explicitly supplied. No overlapping tile inference
is used. The auxiliary skeleton sweep also reads the actual skeleton output,
not the boundary output. Checkpoints save model/optimizer/architecture arguments
and the prior-ramp buffer; evaluation refuses incomplete new-module weights.

`anchor_connections.csv` records loss, positive/negative counts, directed
candidates, mean confidence, actual residual magnitude, ramp and measured
training batch time. `epoch_losses.csv`, `last.pth` and `best.pth` retain their
existing roles. Main train logs reside one level above the run directory to
avoid triggering the trainer's automatic `_2` directory suffix.

## Limits and validation

There is no assertion that past errors were caused by incorrect node relations.
The connection head is an explicit testable constraint, not proof of improved
topology. The 32-anchor budget can miss gaps. Straight narrow corridors cannot
cover long curved roads; they limit allowed learned corrections rather than
claim to reconstruct the true path. A full benchmark, paired on the same GPU,
images, precision and surface threshold, is needed before claiming speed or
clDice/APLS improvements. Candidate counts alone are not an acceleration metric.

Focused tests: `python -m unittest discover -s tests -p test_supervised_anchor_topology.py`.
They cover excessive GT detours, unmatched pairs, empty masks, exact cache
reload, bounded candidates, symmetric logits, gradients, checkpointed ramp,
writeback location/confidence, and empty-candidate identity/finite output.
