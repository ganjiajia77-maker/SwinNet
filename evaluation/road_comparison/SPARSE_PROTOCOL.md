# Sparse official graph evaluation (v5)

This protocol evaluates the same native binary masks and validation-selected
thresholds as v4, but uses fewer graph samples. Its graph scores must be
reported as v5 and must not be mixed with v4 scores.

| Quantity | v4 | v5 |
| --- | ---: | ---: |
| TLTS requested paths per image/model | 1000 | 200 |
| APLS midpoint/control spacing | 50 px | 100 px |
| Sat2Graph TOPO seed density | 0.00050 degrees | 0.00100 degrees |
| Sat2Graph TOPO walk step | 0.00005 degrees | 0.00010 degrees |

The authors' TLTS loop uses `i > n_paths`, so it evaluates one more valid path
than the requested number. All other matching tolerances, the graph conversion,
the reference metric functions, and the validation threshold selection remain
unchanged. APLS control point counts and TOPO seed counts still depend on each
image; there is no 64-node APLS cap.

`compare_predictions.py --workers 2` distributes complete images among two CPU
processes. Each process prepares the ground-truth graph once for its image and
evaluates every model against it. It writes one atomic JSON file per completed
image under `output_dir/partial`. `--resume` skips a completed image only if
the saved protocol, source code, manifest, selection, and its input file hashes
still match. An interrupted image is recomputed. Use the same output directory
and files for resumption; use a new output directory for v5 rather than the v4
directory. Final per-image CSVs and comparison CSV are written after all images
finish. Timing lines identify GT graph construction, clDice/mask work, predicted
graph construction, TOPO, APLS, and TLTS.

```bash
python -u evaluation/road_comparison/compare_predictions.py \
  --manifest comparison.json --split test --selection val_selection.json \
  --output_dir test_sparse_v5 --workers 2 --progress_every 1

# After an interruption, run the same command with --resume appended.
```
