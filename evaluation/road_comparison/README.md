# Unified road mask and raster graph evaluation (v2)

The evaluator accepts the same native-size binary masks and manifests as v1.
Validation threshold selection still uses global IoU only; topology is computed
after that selection, on the chosen validation or test masks. No training or
model inference is required to re-evaluate saved masks.

## Corrected metric names

| Output column | Meaning |
| --- | --- |
| `cldice`, `cldice_precision`, `cldice_recall` | Hard Zhang–Suen skeleton / road-region overlap. |
| `topo_f1`, `topo_precision`, `topo_recall` | **Raster-graph local reachability**. A seed's graph-distance neighborhood is compared with the matched neighborhood in the other graph. These are no longer aliases of clDice. |
| `apls_gt_to_pred`, `apls_pred_to_gt`, `apls_bidirectional` | Sampled control-point shortest-path similarity on pixel graphs. Matched control points are projected to and inserted into the other graph's edges. |
| `tlts_infeasible` | Fraction of connected GT control-point pairs for which the prediction has no matched traversable path. |
| `tlts_unmatched_pairs`, `tlts_disconnected_pairs`, `tlts_matched_pairs` | Explain whether infeasibility came from missing endpoints or graph disconnection. |
| `apls_*_approx` | v1 sampled pixel-node proxy, retained for comparison and explicitly marked approximate. |

Every mask is skeletonized by the same zero-padded Zhang–Suen routine. Each
skeleton pixel is a graph node; 8-neighbor edges have Euclidean pixel lengths.
The raster graph is planar under this convention, so adjacent diagonal pixels
can form connections absent from a vector road network. A GT binary mask cannot
reconstruct overpass levels or the original annotation's vector nodes.

The default graph protocol uses at most 64 deterministic control points per
image, a 5-pixel edge matching radius, at most 16 TOPO seeds per direction,
300-pixel local geodesic radius, 5-pixel sampling cells and 5-pixel matching
tolerance. These **pixel** values are not Sat2Graph's 300-meter setting.
TOPO uses one-to-one spatial matching of sampled reachable local nodes. APLS
uses the mean directional path score and their per-image harmonic mean.
Infeasible includes a reference path whose endpoint cannot be matched;
`tlts_disconnected_matched_rate` isolates true disconnections among matched
endpoint pairs. Images with no GT control-point path have an infeasible rate of
zero and a reported `apls_gt_pairs` count of zero.

The report stores all settings, image hashes, and SHA256 hashes of the metric
implementation. IoU/F1/precision/recall use global TP/FP/FN. Graph and clDice
scores use per-image means; pair counts in the summary are also per-image means.
Compare models only with the same `metric_version`, settings and GT masks.

**Publication label:** use "raster-derived local TOPO-F1", "raster-derived
sampled graph APLS" and "TLTS infeasible under raster control-point matching".
Do not call these the official Sat2Graph or SpaceNet benchmark scores. The
official implementations require graph inputs and make different seed, distance,
coordinate and control-point choices. In particular, this TOPO implementation
uses deterministic pixel seeds and 5-pixel spatial cells, rather than the
official geographic-coordinate sampling protocol.

Sources: [Sat2Graph paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690052.pdf),
[Sat2Graph metric code](https://github.com/songtaohe/Sat2Graph/tree/master/metrics),
[CosmiQ APLS](https://github.com/CosmiQ/apls), and
[TLTS definition and limitations](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730698.pdf).

## Re-evaluate saved masks

Use the *existing* `comparison.json`, `val_selection.json`, and test surface
mask folders from the experiment. The evaluator accepts v1 selection hashes
when the image size and v1 protocol settings match; it does not select a new
threshold on the test set.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
REPO=/home/gjj/SwinNet-h0-encoder-no-a1a2-mass1500
AUDIT=/path/to/existing/unified_eval
python -u "$REPO/evaluation/road_comparison/compare_predictions.py" \
  --manifest "$AUDIT/comparison.json" \
  --split test \
  --selection "$AUDIT/val/val_selection.json" \
  --output_dir "$AUDIT/test_graph_v2" \
  --progress_every 10
cat "$AUDIT/test_graph_v2/test_comparison.csv"
```

`--output_dir` must be new or empty. For a custom three-model manifest, use the
same command and its own saved validation selection. Case IDs must match the GT
exactly; the evaluator rejects missing, extra or duplicated masks. It never
silently resizes an image.
