# ccc7992 global anchor coverage of fixed validation gaps

This independent diagnostic starts at `ccc7992`. No model parameters, training
targets, inference rules, or node-selection rules are changed. A gated recorder
copies the *actual* topk candidates and FPS selections inside
`KeypointGuidedGlobalTopology._extract_fps_anchors`; it does not rerun topk/FPS to
guess the selections. The optional recorder adds no state-dict entries.

## Inputs and forward

- Original fixed198/14-validation-image manifest, original IDs, flattened native
  pixels and anchor groups. Its P0 reference threshold remains0.45.
- Existing ccc7992 checkpoint, requiring the repaired sampling protocol marker.
- Existing evaluation directory containing `protocol.json`, `val_selection.json`,
  `val_probabilities/*.npy` and `val_masks/*.png` for exactly the same14 cases.
- The current final-output threshold comes only from that validation selection
  (e.g.0.48), bound to the checkpoint SHA. No new threshold is fitted.

One validation replay uses the same native1500/512-window/256-stride FP32,
no-TTA, tapered-logit blend minimum0.1 protocol. Only RGB is passed to the model.
Candidates and final nodes use the exact window-pixel coordinates of the current
global module; native coordinates are `(window_top+y, window_left+x)`.
For the Stage2 relocation, one node spans a 4×4 native cell. The diagnostic
uses its center representative (offset + 4 × grid index + 2) and stores both
the original grid coordinate and the native representative in the node table.
The optional `--baseline_evaluation_dir` compares closure and the existing
off-road bridge proxy at the baseline's validation-selected threshold, using
both caches and the same GT. The `baseline_disconnected_common` cohort fixes
the reference gap population for node-coverage comparison. The usual coverage
cohorts retain each checkpoint's own validation-selected threshold.

The real pre-global candidate score remains normalized H norm times the module's
surface prior, followed by topk256 (or the actual configured higher node budget),
score>0 validity, then the original FPS. Invalid zero-padded slots never count.

The first window is checked for bitwise-identical output with and without capture.
All14 stitched probabilities must match the existing evaluation cache to1e-6 and
all thresholded masks must match exactly. Weights/buffers are hashed before and
after. Cached masks and source files are never overwritten. There is no training,
test49 replay, topology recomputation or GT-fed model forward.

## Three primary flags per gap

1. `candidates_both_geometric`: distinct valid actual candidates within the
   fixed native tolerance of each original anchor group in the SAME window.
2. `selected_both_geometric`: the same distinct-node test for final valid FPS nodes.
3. `selected_correct_pair`: two distinct final nodes in the SAME window passing
   the conservative local-GT side check, one for each original side.

`candidates_both_correct` is also reported, distinguishing absent correct sources
from their loss during FPS. The node self-attention uses all valid selected
tokens; there is no additional discrete "pair selection" step. Pair availability
does not prove substantial attention, useful feature updates or actual repair.
Selections from different windows NEVER count as a pair, even if their union
covers both sides. `both_correct_only_across_windows` records that separately.
`no_joint_window` identifies cases where no window can even see both GT-side
regions, rather than assigning all such failures to topk/FPS.

## Tolerance and correct-road check

The predefined default tolerance is12 native pixels, configurable only explicitly
via `--tolerance`. It is not tuned from validation outcomes.
The geometry check uses Euclidean distance from the original anchor sets.
The stricter check requires the node pixel to be GT road, within12 pixels of
GT-road geodesic distance (cardinal steps1, diagonal steps sqrt2) to its own anchors, and closer
along GT to those anchors than to the opposite anchors or the missing centerline.
All checks use a local native-GT crop containing the original gap and anchors,
not a global GT component ID. This rejects a spatially close disconnected
parallel road. It also avoids counting the missing center as both sides.

More than two original anchor groups are NOT silently reduced to two. Those
cases, non-GT reference pixels, mixed local GT roads or unavailable/ambiguous
sides retain their IDs and exclusion reasons; GT-validated fields are unknown,
not false negatives. All rates report their valid two-side denominator.
The method cannot resolve road identity at crossings/overpasses already merged
by the 2D GT; such claims need visual/manual checking. GT-side-rejected hits are
not automatically labelled "parallel-road errors".

## Current disconnection versus old closure proxy

`anchor_connected_local`: a predicted8-connected component touches every original
anchor group inside the original gap/anchor bounding box plus16 pixels.
`still_disconnected` is its negation and is the PRIMARY analysis cohort.
The previous closure proxy additionally requires at least0.8 coverage of the
fixed missing centerline pixels; it is separately preserved as `closed_proxy`.
Thus an already-connected gap with poor centerline coverage is not falsely
called "still disconnected". All198 rows and a closed-gap control are retained.

## Outputs

- `REPORT.md`, `summary.csv/json`: tables for all198, still-disconnected,
  legacy-unclosed-proxy and closed-control cohorts. The counts are gap counts,
  not a sum of overlapping window hits.
- `fixed_gap_node_coverage.csv`: every original case+gap ID, connection state,
  exclusion reason, the three primary flags, correct candidate availability,
  window visibility and cross-window-only hits.
- `gap_windows.csv`: actual per-gap/per-window counts and representative valid
  pair native coordinates; window source identity is preserved.
- `nodes.csv`, `nodes/<case>.npz`: exact native coordinates, scores, validity and
  selected-to-candidate indices for candidates and final nodes.
- `per_image.csv`: counts for currently disconnected gaps, and actual learned
  global residual coefficient. A coefficient near zero remains relevant even
  when node pairs exist.
- `cases_visual/*.png`: RGB with GT/anchors and predicted roads plus candidates
  (yellow), selected nodes (cyan), correct pair A/B (green), missing pixels
  (magenta). Each overlay uses ONE best window rather than merging window nodes.
- `protocol.json`, `checks.json`: checkpoint/manifest/source hashes, threshold,
  tolerance, prediction identity and unchanged-model evidence.

Run using `scripts/diagnose_h0_global_anchor_gaps.py`; `--command show` prints a
compact table without pandas. Supply a fresh output directory. An incomplete
run must not be interpreted as a finished report.
