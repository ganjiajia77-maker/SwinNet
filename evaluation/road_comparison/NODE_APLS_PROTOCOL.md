# Node-only APLS evaluation (v6)

This records the v6 protocol. The current v7 benchmark entry point is
`benchmark_apls_fast.py`; see `FAST_SAMPLED_APLS_PROTOCOL.md` for its commands.

The four-model H0 comparison uses the same binary masks, validation-selected
thresholds, TOPO settings, TLTS settings, and clDice implementation as v5.
APLS alone changes its control-point protocol:

- Every node in the contracted skeleton graph is a control point. Most are
  endpoints or junctions. Ring-only components retain the three synthetic
  anchors required by the graph conversion.
- No control points are inserted along graph edges before matching. The
  existing CosmiQ nearest-edge matching and APLS path scoring functions still
  insert controls from one graph into the other for comparison.
- The ground-truth native graph and all its control-to-control shortest paths
  are prepared once per image and reused for each model. Each prediction still
  has its own native and matched path tables.
- For matched graphs, only shortest paths between control nodes are stored;
  the pinned APLS scorer reads no other paths. Scores match the pinned
  `make_graphs()` implementation with midpoint creation disabled, up to
  floating-point rounding. This is checked by a reference-equivalence test.

This is a new APLS protocol, `road-mask-v6.0-official-node-apls`. Its APLS
scores must not be compared directly with v5 midpoint scores. All four models
must be recomputed under v6. Long road sections with no endpoint or junction
inside them have fewer checks; this can miss local problems between controls.
The scores use graphs reconstructed from raster masks and are not equivalent
to a vector-ground-truth benchmark score.

Each image prints `APLS_GT_NATIVE` once, then each model prints its APLS total,
GT and prediction control counts, prediction native paths, control insertion,
matched paths, and scoring time. The four-model comparison continues to write
per-image partial results and supports `--resume` in a new output directory.

The v6 commit `b302334` retains its original v5-versus-v6 benchmark script.
Use the v7 benchmark to compare v6 full controls, optimized full controls,
and sampled controls on the same saved masks.
