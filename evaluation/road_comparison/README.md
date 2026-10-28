# Unified road evaluation: four topology metrics (v4)

This evaluator accepts saved native-resolution binary surface masks. The
validation threshold remains selected by global IoU; topology runs only after
selection. It does not retrain or re-infer any network.

## What the scores actually call

| Output | Implementation |
| --- | --- |
| `cldice` | The existing hard Zhang–Suen skeleton / road-mask overlap. |
| `topo_precision`, `topo_recall`, `topo_f1` | [Sat2Graph](https://github.com/songtaohe/Sat2Graph) `TOPOGenerateStartingPoints`, `TOPOGeneratePairs`, `TOPOWithPairs`, `TOPO121`, `topoAvg`. Its one-to-one matching and coverage factor are used. |
| `apls_gt_to_pred`, `apls_pred_to_gt`, `apls_bidirectional` | [CosmiQ APLS](https://github.com/CosmiQ/apls) `make_graphs`, `path_sim_metric`, `compute_apls_metric`. The last value is its harmonic bidirectional output. |
| `tlts_correct`, `tlts_too_long`, `tlts_too_short`, `tlts_infeasible` | [Citraro et al.'s released TLTS implementation](https://github.com/lcit/metrics_delin/tree/master/path_based) `toolong_tooshort`. It independently samples GT paths and classifies each by endpoint correspondence and relative shortest-path length. |

The sources, exact upstream commits, untouched downloaded files, licences and
SHA256 checksums are in `vendor/UPSTREAM.json`. Sat2Graph's Python 2 source is
ported to Python 3 by `vendor/build_sat2graph_port.py`; SVG rendering is omitted,
while matching, walking, seed generation and scoring are preserved. CosmiQ's
original function bodies are compiled directly from the saved source by the
adapter; unrelated GDAL, plotting and OSM loaders are not imported.
The TLTS function and its graph-sampling helpers are loaded from the authors'
pinned source. A NetworkX 3 `NodeView` compatibility fix is applied only to a
fallback random-node call. Its score thresholds, path selection and category
rules are unchanged.

Both GT and predictions use the same zero-padded Zhang–Suen skeleton and
8-neighbour pixel graph. For APLS, degree-two chains are contracted to exact
polylines with original pixel-path lengths; there is **no 64-point cap**.
CosmiQ's midpoint insertion, edge projection, path matching and score are run
with a 50-pixel control spacing, 5-pixel snap radius, and 10-pixel minimum
reference path. Sat2Graph's own `main.py` settings are used: 50-metre seed
spacing, 5-metre marble/hole interval, approximately 11-metre match threshold,
and 150/300-metre local graph radius according to its image extent rule. Raster
pixels are mapped to its synthetic one-metre coordinates. The old manifest field
`apls_max_nodes=64` is accepted solely to read existing manifests and has no
effect on v4 scores.

TLTS uses the same full-resolution eight-neighbour skeleton graph, with the
authors' default 1,000 requested GT paths, 25-pixel endpoint match radius and
5% length tolerance. A fixed base seed is combined with the GT skeleton so
each model is scored using the same deterministic GT path sample. The authors'
released loop takes 1,001 paths for `n_paths=1000`; this is retained and
documented for reproducibility. All four TLTS category fractions are saved per
image and averaged across images. `tlts_infeasible` is the headline downward
metric. Empty GT road graphs make TLTS undefined and fail explicitly. An empty
predicted graph is scored as 100% infeasible by the released function.

**Reporting boundary:** These are official *algorithm functions applied to
graphs reconstructed from raster masks*. The dataset supplies no original
vector GT graph or overpass levels. Thus the resulting numbers are not the
published SpaceNet or Sat2Graph benchmark scores. Report graph extraction,
coordinate scale and parameters alongside the result. Existing v1/v2 numbers
must be recomputed for every model before comparing with v4. The TLTS reference
implementation samples raster-reconstructed graphs here, so this is still not
a score against the original vector GT.

## Install and re-evaluate saved test masks

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
METRICS=/home/gjj/unified-road-official-v4
AUDIT=/path/to/existing/unified_eval
python -m pip install -r "$METRICS/evaluation/road_comparison/requirements.txt"

python -u "$METRICS/evaluation/road_comparison/compare_predictions.py" \
  --manifest "$AUDIT/comparison.json" \
  --split test \
  --selection "$AUDIT/val/val_selection.json" \
  --output_dir "$AUDIT/test_official_graph_v4" \
  --progress_every 1

cat "$AUDIT/test_official_graph_v4/test_comparison.csv"
```

Use the existing manifest and threshold selection together, so checkpoint
identity and saved mask directories remain paired. The output directory must be
new or empty. The evaluator rejects missing, extra, duplicated or resized masks.
The full official APLS workflow computes all-pairs graph paths and can take
substantial CPU time or memory on complex 1500-pixel masks. Per-image CSV rows
are flushed immediately so progress is visible.
