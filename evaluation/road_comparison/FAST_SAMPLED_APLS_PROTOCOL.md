# Fast sampled node APLS (v7)

This version changes only APLS. The saved binary masks, validation thresholds,
clDice, TOPO and TLTS settings remain those of v6.

## Exact faster mode

Set `"apls_max_control_nodes": 0` in the manifest protocol to retain every
original compact-graph node. The original CosmiQ node insertion and APLS
scoring functions remain in use. The adapter finds the incident edges of the
same nearby-node set directly instead of enumerating every graph edge for each
control point. It reproduces the original NetworkX edge orientation and the
first-edge rule when distances tie. The benchmark compares all four v6 and
fast-full scores and raises an error if any differ by more than `1e-10`.

## Sampled mode

The default protocol selects at most 500 original compact-graph nodes from
each GT and prediction graph. It uses a local RNG seeded by SHA-256 of that
graph's raster skeleton coordinates and `apls_control_sampling_seed` (default
1234). The GT control set is generated once per image and reused for every
model. Prediction controls are sampled independently using the same rule.
Graphs with 500 or fewer nodes use all nodes. Selection occurs after the
shared graph conversion, so the graph paths still run along full polylines.

The control count can be changed through `manifest["protocol"]`:

```json
{"apls_max_control_nodes": 256, "apls_control_sampling_seed": 1234}
```

`0` means all nodes. Values such as 256, 500 and 1000 define different APLS
sampling protocols. Do not combine their scores or compare them directly with
v6 full-node scores. Each protocol needs a separate output directory and all
models must be evaluated with the same GT sample and settings. A small cap can
miss local defects or alter close model rankings; validate the chosen cap on
representative images and several seeds before treating it as final.

The underlying path score is still the pinned CosmiQ implementation applied
to graphs reconstructed from raster masks. Sampling here is a documented
large-graph adaptation, not a claim of equivalence to vector-GT benchmark
scores.

## Same-image timing and score comparison

```bash
python -u evaluation/road_comparison/benchmark_apls_fast.py \
  --manifest comparison.json --selection val_selection.json \
  --case 10378780_15 --sample_controls 500 \
  --old_evaluator /path/to/v6/evaluation/road_comparison/official_graph_metrics.py
```

This runs v6 full nodes, v7 fast full nodes, and v7 sampled nodes on the same
prebuilt graphs. The output separates shared GT, prediction native paths,
control insertion, matched paths and scoring. It excludes disk I/O,
skeletonization, TOPO and TLTS. Full test evaluation remains resumable, but
v7 must use a fresh output directory because its protocol and code hashes
differ from v6.
