# Validated selector -> joint actual transport

The original source-selector forward is reused without modification: eight
actual neighbours plus stay, three newly computed states, original rate budget,
proposal and anchor, and no extra scalar G. This entry point does not rerun
relationship pretraining and does not automatically retrain B_control.

Before loading GPU models, read the supplied selector experiment.json,
relation_best/relations.json, and relation/best.pth metadata. Match phase/config,
report score, code hashes, identity and original source SHA256. Locate original
V9 by hash and matching val-selection metadata, not a timestamp. Verify every
original module in the relation checkpoint is still identical to that original
V9. Frozen BASE/Q3 must still exist with their exact saved hashes.

Find an existing B_control using config and last/best checkpoint metadata:
same identity, source hash,20 epochs, source learning rates/losses, original
Stage-D optimiser scope, seed, physical batch2, accumulation1, workers, FP32,
no EMA and val selection cadence. The original L_G-config field may be present
in that control artifact, but variant=control, recorded total losses and the
recorded gradient flag must prove it was NOT included in optimisation. Reject
incompatible/ambiguous controls instead of silently starting another training.
Every new epoch must match the control's FULL case-order and augmented-image
tensor hashes. Reproduce the existing control's saved val TP/FP/FN at.60.

S_selector starts with ORIGINAL V9 modules plus the validated selector state.
Jointly optimise selector and the active visual/R/W/rates/B/A/projection/pair
modules. Shared modules use their original B_control LR groups; the new selector
uses its validated pretraining LR. BASE/Q3 remain eval/frozen. The unused old
state_gate and semantic-G delta layers are excluded. Split the original9-output
direction head into a frozen old-G row and eight trainable W rows, so even AdamW
decay cannot move that unused row. Save/load this head in canonical original
shape with strict checking; no original core file/hash is changed.

Loss: Lseg+.2Lcont+lambda_relation*Lrelation. Reuse the validated native labels,
interest rules, image/step/direction/sign/magnitude/source matching, class balance
and equal-three-step normalisation. Use LIVE source_edge_logits from the NEW
trajectory; labels/matching alone detach. There is no second frozen-trajectory
forward for the training loss, no L_G and no four candidate path losses. The
joint relation coefficient comes from an explicit prior config if available,
otherwise initially0.1 and is marked experimental. Log counts, losses and the
weighted relation/segmentation gradient norm ratio. Empty-label epochs fail.

One physical-batch2 precheck records selector perturbation -> transport/P1,
segmentation gradients -> selector, new-U inputs at steps2/3, probability budget,
invalid-neighbour masking, stay capacity and scalar-G independence. GT ignores
affect only supervision/statistics. No native GT relation mask enters inference.

Freeze validation references BEFORE the first continuation epoch. If the prior
diagnosis only saved the P0/GT/skeleton fingerprint, materialise its identical
206 gaps and native anchor pixels ONCE using the unchanged old definition,
after matching that existing fingerprint. All later checkpoints/models consume
that immutable gap/anchor JSON; no per-model gap/anchor generation is used.

Checkpoint rule at val.60 (same cadence as existing control), and final val
threshold rule: IoU+.15clDice+.10APLS_GT_to_Pred+.05APLS_bidirectional
-.001short_components-.01bridge_proxy/images. Scan.35:.01:.75. Control threshold
curves can be reused only after exact prediction/GT cache correspondence;
otherwise recompute metrics without retraining. Curves resume per threshold.
Matched Recall uses exact positive-score boundaries and records unavoidable
tie residuals. Report IoU/F1/precision/recall/clDice/bidirectional approximate
APLS/fixed closures/bridge proxies/short fragments at own and matched thresholds.

actual_source_relations.csv records actual per-source weights and transport
into supported/blocked/uncertain native relationships. Preserve the ORIGINAL
fixed-input relationship test before/after. Joint encoders may drift, so loss of
fixed-input compatibility is not by itself proof of a worse gate classifier.
No test split is read. All topology quantities remain proxy measures. Selector
pretraining is extra recorded supervision; this is not a pure loss-only ablation.

`bash scripts/run_v9_selector_transport.sh all` performs metadata preflight,
actual update check, fixed-reference preparation,20 joint epochs, val sweep,
paired comparison and relationship diagnostics. It resolves the supplied
20261030_110206 run by default. `preflight`, `smoke`, `train`, `val`, `results`
are available; optional CONTROL_RUN/ORIGINAL_RUN/REFERENCE_SET disambiguate
existing artifacts. Exact optimiser/RNG/config checkpoints permit resuming.
