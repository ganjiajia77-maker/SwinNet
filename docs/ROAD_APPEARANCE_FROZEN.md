# Training-only road-directed shadow and texture: frozen S_selector

This implements the user's latest **frozen BASE/Q3** choice, starting from the
trained actual_transport S_selector **before L_choice**. It is a separate
experiment from slim end-to-end. No architecture or existing loss is changed.
No L_choice/L_G/path loss is called. The complete objective remains the source
S_selector's L_seg +0.2 L_cont +0.1 L_relation (coefficients loaded from metadata).

## Exactly what is frozen / recomputed

Strictly load the original S checkpoint and exact trained BASE/Q3 dependencies.
BASE/Q3 stay eval, requires_grad=False, no_grad. The same original trainable
selector, visual, R, W, rates, B, A and active pair-feature parameters use the same
optimizer groups/LR/weight decay. Unused scalar-G parameters remain excluded.
**Augmented RGB goes through BASE/Q3 afresh for every training sample.** No clean
P0/H/H3/Q cache is consumed. In clean val, frozen P0 should be exactly identical
in the augmentation-off/on variants; the paired analysis verifies it.

## Controlled data pipeline

The old shared loader actually geometrically/color-augmented the full1500 image
before cropping. It is left untouched for existing experiments. This new loader
uses the requested **crop512 -> same original geometry function -> same original
color/degrade function -> new local perturbation -> ImageNet normalization**.
Skeleton targets are generated in the final crop coordinate system. Off and on
share this exact pipeline. Hence an OLD full-image-augment-first run is not a
strict augmentation-only control. New off/on controls are paired from the same
trained S checkpoint, at the same budget. Original deterministic crop_position
and geometry RNG seed/draw sequence are reused; only the stage order is made
explicit and identical for this new pair.

Only RGB changes. GT, skeleton and dilated skeleton are unchanged by local
appearance augmentation; debug masks never enter the actual image tensor.
Training explicitly calls model(batch['image']); class/tangent/center metadata
is logged separately and never passed to the model.

## Sampling and finite failure policy

AppearanceConfig stores all settings. Each tile exclusively requests:50% no
local perturbation,25% one shadow,25% one texture. Of perturbation requests80%
target roads and20% nearby background. Actual success can be lower and is logged.
No target/type switching or black/solid fallback occurs on failure.

- Road points: native crop GT skeleton, exclude endpoints/junctions within8px,
  initial border margin16px; local PCA radius10, minimum6 skeleton pixels,
  principal-eigenvalue fraction>=.85. Final rectangle must fit with8px context.
- Road width:2*GT distance transform at the selected reliable skeleton point.
- Direction:GT tangent +/-15degrees; center jitter +/-2px. Smooth rotated
  rectangles are independent of the GT road contour.
- Road rectangles must leave GT road AND skeleton support in both unmasked
  along-axis context disks. This verifies unmasked GT context, not that the
  original RGB is semantically free of natural occlusion.
- Background centers:6..64px from GT road, with nearest reliable skeleton
  orientation; final rectangle GT road fraction<=5%.
- Shadow along-road length16..64, cross-road width1.5..3 times estimated road
  width capped16..96, transition2..4px, RGB alpha0.35..0.75. No extra blur/color.
- Texture along-road length8..32, cross-road width1.5..2.5 times road width
  capped12..64, transition1..2px. Core is exact full donor-pixel replacement.
- A same-tile, unresized, axis-aligned donor patch covers the target bounding
  box, has<=1% GT road, and its bounding box does not overlap the target box.
  Spatial std below2 rejects blank/solid donors; there is no generated texture.
- Placement attempts<=10. Once a valid texture target is selected, donor
  attempts<=10; failure skips this perturbation, not a nested100-target search.

The requested dimensions can otherwise make the cross-road width exceed the
along-road length (e.g. width96 vs maximum shadow length64). To respect the user's
long-side-along-road requirement, such draws are rejected within the same10
placement attempts. Thus wide roads may be skipped; the reason
width_exceeds_along_road_length is explicitly counted rather than silently
rotating the wrong side or changing width factors. This is an initial sampling
choice, not a validated optimum.

## Randomness and monitoring

Independent PCG64 seeds derive from total seed, absolute crop epoch, sample ID,
crop ordinal and namespace. The local RNG never consumes base crop/geometry/color,
global NumPy or Torch randomness. Metadata includes pre-local RGB hash, label
hashes, geometry tuple and crop coordinates. Pair comparison checks all epochs'
case order and pre-local input hashes before reporting results.

`appearance_epoch_*.jsonl` records each tile's request, target, success, bounded
retry reasons, dimensions, mask area fraction, alpha, donor purity/location and
read/augmentation time. `epoch_*.json` aggregates requests/success/skips, actual
road/background fractions, distributions, data-wait and training-iteration time.
Decode time and dataset-preparation time are distinct. Nothing is inferred from
the forced preview proportions.

## Preview, training and comparison

    bash scripts/run_v9_road_appearance.sh preview
    bash scripts/run_v9_road_appearance.sh pair
    bash scripts/run_v9_road_appearance.sh results

Preview exports24 successful TRAIN-only examples:8 road shadows,8 road textures,
4 background shadows and4 background textures. Each shows pre-local RGB, local
augmented RGB, unchanged GT/skeleton and diagnostic-only mask. PNGs, metadata and
an offline page are saved, then tarred as preview24.tar.gz. If all24 cannot be
collected within a finite1108-sample preview budget, training stops with reasons.

`pair` reuses/validates that preview, trains off then on from the SAME original S
checkpoint, evaluates clean val, saves P0/P1 probabilities and compares own val,
common, matched Recall and matched foreground-area points. No prompt/Enter is
required. Test is a separate final action: `...sh test` uses val thresholds only.

The epoch budget, LR, seed, workers and val interval come from the source S config
(the current source normally20epochs). Generic EPOCHS env vars are ignored.
APPEARANCE_EPOCHS is an explicit optional override, applied equally to both groups.
No old relationship pretraining is repeated. Last checkpoints resume model,
optimizer and RNG; completed epochs are skipped with an explicit RESUME message.
Because the comparison adds continuation training, reusing the untouched old S
as the ONLY control would confound augmentation with extra training.

`on` runs only the new augmented continuation, `off` only its matched control.
On/off aliases automatically resume their own directory. Source weights and
existing experiment directories remain untouched. A new PAIR_DIR is required
to change the sampling configuration or training budget.

The real startup smoke uses two augmented training crops. It hooks BASE/Q3 to
verify they read the perturbed tensor, verifies eval/no-grad, measures changed
P0/Q vs clean input, and checks repair gradients. It does not update weights.

Local tests: exclusive request probabilities; bounded retries; road/background
constraints; exact donor core replacement/no overlap; no solid fallback; seed
replay/global RNG isolation; identical off/on base crops/geometry/color and GT;
24-example export; real downstream frozen forward and absent L_choice call.
Actual server success rates, timing and scientific benefit remain unmeasured
until run there. APLS/closure/bridge are raster proxy metrics.
