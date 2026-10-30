# Fixed-action acceptance diagnosis

This is a read-only diagnostic of the completed equal-budget B_control/C_gate3
continuations. It starts no training, modifies no checkpoint or training source,
and uses the same14 val images with all512/256 windows. The three hashed training
files from8ed6b6b are unchanged, so that training experiment can still resume.

The earlier comparison was on-policy: each model produced different states,
sources and transport actions. This diagnostic instead captures two banks,
one from B and one from C. Within each bank, both gate scorers judge exactly
the same actual U/M/V/rate/proposal/anchor and frozen feature inputs. Validity,
BCE-benefit labels and signed V/magnitude matching are computed only once.
No Oracle gate is substituted into state updates.

The primary full_gate comparison re-evaluates each scorer's visual projection
and relations on the same bank raw H/H3/P0/R. This keeps each model's visual
coordinate system consistent with its trained G. Internal relation evidence/W may differ as a G input,
but it never changes physical M/rate/V/state or action labels. A secondary
residual_only comparison also fixes bank old-G logit, W, two and one; it isolates
the learned semantic/state corrections on a common visual embedding. These
secondary cross-model common embeddings can be off-distribution and should not
be read as standalone proof of defective layers. Self replay must reproduce the bank's
actual gate logit before rows are accepted. This is gate replay, not Surface
counterfactual inference and not a new topology performance claim.

For both natural and same-image/step/sign/V/action matched populations, report
harmful acceptance at helpful acceptance50/70/80/90/95%, and at B's original
G>=.5 helpful rate. Scores are unsquashed logits to retain saturated-score order.
At a tied boundary, use one identical random acceptance fraction for BOTH
classes: the matched rates are exact expectations. Include deterministic
strict/inclusive bounds rather than claim an unattainable deterministic match.
Thresholds are diagnostic only; the trained soft G is not changed.

Results include full/per-step/action-sign ROC/PR-AUC, matched rates, per-image
differences, and paired-image bootstrap intervals. A uniform monotonic score
shift yields identical matched curves; lowering all G cannot pass as improved
ranking. Both action banks must be inspected because each remains conditional
on its generating model. Helpful/harmful labels are local BCE/clear-GT proxies,
not perfect GT topology labels. Better replay ranking alone does not establish
better actual gap closure, bridges or final P1.

Each image/bank saves an input-hashed, checksum-verified NPZ tape with the shared
action/label/matching rows, native tile and work-grid coordinates, U/M/rate/V,
off/on and four scorer-logit vectors. No independent resampling occurs between
B and C. Full caches allow `cached` analysis without upstream model loading;
`run` still requires the exact frozen BASE/Q3 checkpoints from original V9.

Run `bash scripts/run_v9_fixed_actions.sh run`, then `results`. The launcher
resolves the completed paired experiment from latest_v9_gate_benefit_run.txt;
an explicit GATE_BENEFIT_RUN overrides that path. It rejects missing/incomplete
or nonpaired continuations instead of silently training a missing control.
Reports go to that experiment's fixed_action_equal_helpful_v1 directory.
