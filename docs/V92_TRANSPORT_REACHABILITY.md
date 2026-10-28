# V9.2 local oracle and training path audit

This is the `direct_random_downstream` V9 run, not a strict ablation of the
original trained V8. It loads the V9 checkpoint and the exact BASE/Q3 files
recorded in its identity; model weights and original forward code stay intact.
It is a diagnostic only and does not start a new training run.

## Local greedy Oracle G

On the 14 validation images, the oracle recomputes the original V9 dynamics at
each of three steps. It calculates the pre-gate action
`rates[:,1] * (M - U)` and compares the local BCE after setting G to 0 versus
1 while holding that step's proposal and anchor increments fixed. GT road and
GT background guide that choice; mixed labels, tiny actions and near-tie gains
keep the original G. Negative actions are judged by the same loss difference.
At the next step all states, source support and rates are recomputed normally.
GT is available only to this diagnostic intervention.

This is a **local greedy diagnostic**, not a mathematical upper bound for the
whole network's APLS, closure or false bridges. A local probability correction
can affect later states in either direction. Window-level decision counts
include overlap; pixel metrics are computed on stitched full validation images.
The output compares original and Oracle at the original val threshold, at the
previously discussed fixed threshold 0.35, matched Recall and matched foreground
area, reporting matching residuals. It checks the
P0/GT/skeleton hash against the successful V9.1 reference set and requires 206
eligible reference gaps. APLS and closure/bridge numbers retain V9 proxy rules.

## Training path coverage and proposal audit

Every native training image is scanned with 512 windows and stride 256. For
each tile, positive pre-gate candidates require a positive update over 1e-5
and source mass over 1e-6. Coverage reports any-hit, per-gap coverage, full
coverage and longest no-candidate run on simple GT skeleton paths. Branching,
loops and tile-margin gaps are separately identified. The three-step directed
graph requires an initial P0/R anchor, nonzero source support, positive
pre-gate candidate and a selected W direction. Defaults `R>=0.1` and
`W_direction>=0.01` are diagnostic thresholds, not model rules. Directional
reach is computed from maps of the original run, constrained to the GT path
corridor, and is not an intervention rerun or proof of actual closure.
The P1 binary closure proxy is recorded separately.

P0 thresholds 0.30/0.45/0.60 generate predicted skeleton endpoints, pairs of
different predicted components, and single-sided extensions. P0, R, W, Q and
predicted local geometry are used to construct proposals and classifier
features. GT is applied afterward to snap endpoints to a GT skeleton, check a
bounded GT shortest path, label clear cases, and flag ambiguous cases. Parallel,
lateral, isolated and extension categories are heuristic proxies, not certified
semantic classes. Work-grid GT downsampling can merge thin or nearby roads.
The candidate cap is recorded; recall includes its effect.

Each tile's gaps and proposals use first-window centroid/midpoint ownership to
reduce duplication. Because P0 and dynamics are computed independently per
window, a path crossing a tile edge cannot be completely assessed, and gap
counts are not a perfectly deduplicated stitched 1500-pixel census. The CSV
keeps boundary-excluded rows; simple-path reach rates use their own denominators.
Candidate positive GT-path overlap and local reachable coverage are distinct
from actual binary closure. The GT path overlap uses one work-pixel tolerance.

The lightweight logistic diagnostic classifier uses only candidate features.
Image IDs are deterministically split inside the **training** set; all windows
from an image remain in one split. PR-AUC/ROC-AUC on the held-out training
images are reported with natural class prevalence and with matched sampling
by proposal threshold, length and type. Memory-bounded per-class reservoirs
are used; natural PR-AUC is prevalence-weighted when sampled and is an
estimate. Neither val nor test supplies training examples. Uncertain labels
are excluded and their counts remain visible in the CSV and JSON.

## Running

Activate the original server environment and run `scripts/run_v92_mass.sh
oracle`, optionally `scripts/run_v92_mass.sh train_smoke` for one training
image, then `scripts/run_v92_mass.sh train_paths` for the full split. The
one-image result is marked `complete_train_split=false` and must not be used
as a dataset-wide result. The oracle command
reads `latest_v91_root.txt`, then `latest_counterfactual.txt`, to find the
fixed reference fingerprint. Set `REFERENCE_SET` explicitly if those pointers
were moved. Full training image scanning is much slower than the 14-image
oracle and can produce a large CSV; progress is logged per image. Each rerun
creates a new directory and updates its latest-success pointer only on success.

Interpretation requires both reports: improved matched-Recall tradeoffs with
Oracle G indicate local gate potential; a small change or poor three-step
reach suggests sources, W, M and range may be limiting. Neither result by itself
proves a single causal root or warrants choosing a new model on the test set.
