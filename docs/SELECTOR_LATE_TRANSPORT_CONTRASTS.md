# Read-only late transport contrasts

Same current joint-transport checkpoint,14 val images, native512/256 windows,
same frozen P0.45/206 gaps AND anchor arrays, and ONE common Surface threshold.
Default threshold is the current S_selector own-val threshold in comparison.json;
no sweep or per-contrast tuning is performed. Explicit THRESHOLD is supported.
Best-checkpoint epoch/score/version, frozen model identity, code hashes and the
immutable manifest binding are checked before inference. No optimiser exists.

- full_three: ordinary predicted source choices at all three steps.
- stop_late: first step unchanged; at steps2/3 transfer ALL source weight to
  stay. Only transport becomes zero; rate computation, proposal and anchor
  still run on this contrast's new states.
- oracle_late: first step unchanged; at steps2/3 transfer only native GT
  confirmed wrong-source weights to stay. Positive and unknown(-1) choices
  remain EXACTLY unchanged, with no source renormalisation. This is GT-assisted
  diagnostic inference, never deployable performance or a rigorous global bound.

Native labels use the same checkpoint-saved settings as relationship training.
They are local corridor proxies; ambiguous junctions, close-road endpoints and
boundaries are not Oracle-rejected. Both positive and negative updates from
wrong relations are removed; removed_positive/removed_negative_abs show that
Oracle can also discard negative probability corrections. Do not assume every
removed transport was harmful to pixel BCE.

The original source-selector implementation is unchanged. A temporary selector
output hook changes only the nine choices, and is removed even on errors.
Rates, source support, neighbours and choices are recomputed for each new U.
The first-step state/choice equality is asserted on every tile. Frozen backbone
features are shared across contrasts, avoiding three backbone inferences.

REPORT.md / step_comparison.csv show final and each-step FP/FN/IoU/bridge proxy,
fixed closures, newly created/removed FP/FN and gained/lost closures. P0 row uses
the SAME common threshold for step-change interpretation; gap definitions remain
fixed at original P0.45 regardless of that common prediction threshold.

Every image saves full native FP32 probabilities for step1(shared), and each
mode's step2/step3 in *_steps.npz. Representative gap-rich tiles for the first4
images save U0-U3, raw and used choices, transport/proposal/anchor/rates/V in
*_tile_states.npz. FP/FN montage: greenTP/redFP/blueFN. This avoids saving huge
all-tile nine-direction tensors while retaining full-image per-step results.
Per-image caches resume after interruption; identity changes require a new
output directory. Input checkpoint and training artifacts are never written.

Run `bash scripts/run_selector_late_transport.sh run`, then `results`.
The existing actual-transport run marker resolves the correct run directory;
TRANSPORT_RUN can explicitly override it. All topology measures remain proxies.
