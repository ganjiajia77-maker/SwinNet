# Local C3-guided surface contact supervision

Base: `bb26a17`, branch `codex/h0-local-cs-bb26a17`. Network files are unchanged.
No cross-scale path-training commits are included.

## Code mapping

- `datasets/native_relation_targets.py`: unchanged. Targets enumerate native
  eight-neighbour road contacts across work cells; multiple road components in
  a cell invalidate incident relations. Construction follows crop/augmentation.
- `topology_direction_constants.py`: unchanged N, NE, E, SE, S, SW, W, NW.
- `losses/local_cs_loss.py`: derives undirected representatives from offsets,
  not direction numbers. Caches batched native-contact indices, selects maximum
  detached min-endpoint probability. Positive selection is restricted to GT
  road/road pairs; negatives supervise only background endpoints. At stride 4
  cardinal edges have ten crossing pixel pairs, diagonals one. No pixel loops,
  graph search, new labels, crop mining, negative sampling or quotas.
- `losses/road_losses.py`: reuses the native targets already built for C/D and
  adds only `lambda_local_cs * L_local_cs` to the original B objective. C3
  weights are detached. Validation under no_grad excludes this training term.
- `train_image.py`: three loss modes, strict checkpoint-only initialization
  with fresh optimizer/epoch zero, hashes, batch counts, loss/gradient audits.
  No graph structure or prediction output is modified by the mode.

Per image and per class: sum weighted edge losses / number of valid edges.
Missing classes contribute zero; then average images and multiply each class
by 0.5. C weights are not normalized to sum one. Lambda 0.05 is experimental.
Both positive and negative labels that disagree with native contacts raise an
error with mismatch counts; no silent relabeling.

## Short experiment

Requires an explicit original native B checkpoint (not path_joint). All modes
load exactly the same tensors and ImageNet/new parameter group identities.
Original LR settings, warmup, AdamW, augmentations, seeds and 80-epoch schedule
are retained. Defaults run only the first 100 batches of epoch 1 using the
existing `finetune_epochs` stop window. No 80-epoch training starts implicitly.
Use `last.pth` after this short run; original validation is every fifth epoch,
so a one-epoch short run normally has no `best.pth`. Run the separate evaluator.

```
export LOCAL_CS_INIT=/absolute/path/to/original_native_B/best.pth
export LOCAL_CS_PAIR=h0_local_cs_short_$(date +%Y%m%d_%H%M%S)
CUDA_VISIBLE_DEVICES=2 SHORT_EPOCHS=1 SHORT_BATCHES=100 bash scripts/run_local_cs_short.sh all
```

`all` runs off, surface_only, c_guided sequentially. Each can also be invoked
independently. `summarize_local_cs.py` rejects differing initialization tensors,
checkpoint SHA, image order, augmented RGB or actual completed batch counts.
Batch-time comparison excludes first five batches and gradient-audit batches.
This is measured time including existing B native targets; no speedup is claimed
without running all modes on comparable GPU load. JSONL records empty classes,
raw/weighted loss, counts and surface-logit gradient norms (first and every 50th
batch). They are not parameter-gradient norms or a proof of net topology benefit.

## Paired validation

```
python scripts/evaluate_local_cs.py --initial "$LOCAL_CS_INIT" \
  --run "off=$RESULTS/${LOCAL_CS_PAIR}_off/last.pth" \
  --run "surface_only=$RESULTS/${LOCAL_CS_PAIR}_surface_only/last.pth" \
  --run "c_guided=$RESULTS/${LOCAL_CS_PAIR}_c_guided/last.pth" \
  --root_path /home/gjj/Swin-Unet-main/Massachusetts/tiff \
  --fixed_gaps /home/gjj/SwinNet-roadbias/model_out/full_h0_joint_pair_v1/reference/fixed_gaps_and_anchors.json \
  --output_dir "$RESULTS/${LOCAL_CS_PAIR}_val"
```

`relations.csv` fixes relationships on the initial checkpoint: positive/C3>=0.5/
surface<0.2, originally correct positives, and originally correct negatives.
Reports recovery, lost positives, newly accepted negative relations and the
two-road negative subset. GT ambiguity and single ownership per sliding tile
are retained. Thresholds here are diagnostic and never act as training gates.

`fixed_gap_changes.csv` preserves gap IDs/anchors, requires 80% native missing
pixel coverage and anchor connectivity within their joint bounding box expanded
by 16 pixels. It is a local gap proxy, not manual verification of correct road
recovery. `comparison.csv` contains same-threshold and own-validation-threshold
IoU/P/R/F1/clDice plus new-bridge proxies. Own thresholds maximize val global F1,
as in the existing native evaluator; test is never used to choose thresholds.

`manifest.json` exports validation masks at those thresholds for the *same*
external six-metric tool used previously (TOPO/APLS/TLTS); local raster APLS is
not reported as graph APLS. Full topology on all four models is optional after
the short test and will add CPU evaluation time. This evaluator performs no
training and no path loss. It does not change training checkpoint selection.

## Checks

Synthetic tests cover native contact equality, all geometric directions via
rotations/flips, non-multiple-of-four source crop origins, ambiguous cells, no
wrapping, GT-only positive gradients, background-only negative gradients,
reselection, detached C weights, no weight-sum normalization, empty classes,
label mismatch failures, batch normalization, original loss preservation in
off mode and additive loss integration. CPU checks do not establish real GPU
throughput or Massachusetts performance.

## Full training from the original B initialization

`scripts/run_native_h0_pair.sh local_cs_fresh` runs only c_guided, from the
original ImageNet initialization, with no trained B weights or optimizer state.
It uses the original common 80-epoch configuration, unlimited batches, and
checks the complete initialization manifest against original B before training.
The manifest defaults to
`$RESULTS/h0_native_relation_80e_s1234_preflight_B/initial_state.json`;
`B_INITIAL_MANIFEST` can specify the original B training run's initial_state.json.
Missing or mismatching evidence stops training rather than silently changing
initialization. Old LOCAL_CS_INIT/SHORT_EPOCHS/SHORT_BATCHES variables are not
used by this mode. Set LOCAL_CS_RUN to a new independent directory name.

```
export LOCAL_CS_RUN=h0_local_cs_guided_fresh80_$(date +%Y%m%d_%H%M%S)
CUDA_VISIBLE_DEVICES=2 bash scripts/run_native_h0_pair.sh local_cs_fresh
```
