# Unified road evaluation: official graph metric functions (v3)

This evaluator accepts saved native-resolution binary surface masks. The
validation threshold remains selected by global IoU; topology runs only after
selection. It does not retrain or re-infer any network.

## What the scores actually call

| Output | Implementation |
| --- | --- |
| `cldice` | The existing hard Zhang–Suen skeleton / road-mask overlap. |
| `topo_precision`, `topo_recall`, `topo_f1` | [Sat2Graph](https://github.com/songtaohe/Sat2Graph) `TOPOGenerateStartingPoints`, `TOPOGeneratePairs`, `TOPOWithPairs`, `TOPO121`, `topoAvg`. Its one-to-one matching and coverage factor are used. |
| `apls_gt_to_pred`, `apls_pred_to_gt`, `apls_bidirectional` | [CosmiQ APLS](https://github.com/CosmiQ/apls) `make_graphs`, `path_sim_metric`, `compute_apls_metric`. The last value is its harmonic bidirectional output. |
| `tlts_infeasible` | Diagnostic on the **same GT control-point routes used by official APLS**: fraction with unmatched or disconnected endpoints in the inserted prediction graph. TLTS is not part of either official library and must be labelled a raster-derived diagnostic. |

The sources, exact upstream commits, untouched downloaded files, licences and
SHA256 checksums are in `vendor/UPSTREAM.json`. Sat2Graph's Python 2 source is
ported to Python 3 by `vendor/build_sat2graph_port.py`; SVG rendering is omitted,
while matching, walking, seed generation and scoring are preserved. CosmiQ's
original function bodies are compiled directly from the saved source by the
adapter; unrelated GDAL, plotting and OSM loaders are not imported.

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
effect on v3 scores.

**Reporting boundary:** These are official *algorithm functions applied to
graphs reconstructed from raster masks*. The dataset supplies no original
vector GT graph or overpass levels. Thus the resulting numbers are not the
published SpaceNet or Sat2Graph benchmark scores. Report graph extraction,
coordinate scale and parameters alongside the result. Existing v1/v2 numbers
must be recomputed for every model before comparing with v3.

## Install and re-evaluate saved test masks

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swinunet
REPO=/home/gjj/SwinNet-h0-encoder-no-a1a2-mass1500
AUDIT=/path/to/existing/unified_eval
python -m pip install -r "$REPO/evaluation/road_comparison/requirements.txt"

python -u "$REPO/evaluation/road_comparison/compare_predictions.py" \
  --manifest "$AUDIT/comparison.json" \
  --split test \
  --selection "$AUDIT/val/val_selection.json" \
  --output_dir "$AUDIT/test_official_graph_v3" \
  --progress_every 1

cat "$AUDIT/test_official_graph_v3/test_comparison.csv"
```

Use the existing manifest and threshold selection together, so checkpoint
identity and saved mask directories remain paired. The output directory must be
new or empty. The evaluator rejects missing, extra, duplicated or resized masks.
The full official APLS workflow computes all-pairs graph paths and can take
substantial CPU time or memory on complex 1500-pixel masks. Per-image CSV rows
are flushed immediately so progress is visible.
