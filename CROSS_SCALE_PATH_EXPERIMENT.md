# Native H0 path supervision experiment

`joint` adds GT-supervised maximin surface/C3 paths and detached, bounded
conflict weights. `off` preserves B. `surface` uses the same GT path losses
with uniform weights; setting its C coefficient to zero gives the surface-only
experiment. For a fair **conflict-weighting** comparison, set the same
nonzero surface and C coefficients in `surface` and `joint`; coefficient
equality does not imply that measured loss or gradient magnitudes are equal.

The reference training entry keeps B's 80-epoch ImageNet initialization,
random crop, augmentation, LR, loss and validation settings, and verifies the
existing B initialization fingerprint. A new run is not a continuation of
B's `best.pth`. This permits direct comparison to already trained B with the
same initialization and 80-epoch budget. Shared-data hashes and checkpoint
identity must be checked; equal initialization does not imply equal trained
weights.

Path samples use augmented native masks, a 4x4 grid, local GT paths, and local
window connected components. Non-connected in a bounded GT window is a
local proxy; it does not establish that roads are disconnected globally or
that overpass labels capture physical travel. Maximin search operates on
detached predictions, then BCE samples the original logits. Native positive C
edges and both directional validity masks must be valid. Background-only loss
on negative paths avoids forcing endpoints off roads. Ambiguous or truncated
cases are skipped and reported. Unrepresentable C negatives retain their valid
surface loss. Candidate pairs are not prediction endpoints and are not a
candidate-discovery or completeness guarantee.

`path_training.jsonl` records requested/used pairs, skip reasons, mismatch
counts, weighted losses and search time for each batch. With
`--path_audit_gradients`, the first valid batch records weighted versus original
output-logit gradient norms in `path_gradient_audit.json`; exceeding the
configurable initial ratio stops the run. This is a pilot guard, not a proof of
balanced parameter-level gradients over training. If no path supervision is
available for a whole epoch, training aborts.

`scripts/evaluate_native_path_pair.py` compares two *native* checkpoints using
the fixed 198-gap manifest and existing validation protocol. It requires both
80-epoch runs and matching fingerprints. The existing graph and gap measures
remain proxies. A B-versus-joint comparison establishes the whole training
change only: without a matched-C `surface` run it does not isolate the
cross-scale conflict-weighting component.
