# H0 native relation experiment

Base architecture: 599a410. `scripts/run_native_h0_pair.sh prepare` compares
shared initial weights and runs a real 512-pixel native-model smoke test. The
`A` and `B` commands train the unmodified H0 control and native-relation H0 for
80 epochs; `resume_A`, `resume_B` and `val` are separate commands. All use the
same input order, crop and augmentation seed, batch size, FP32 and no EMA.
Persistent workers are disabled for both models so epoch-dependent crop seeds
propagate to workers. Initial model hashes and per-epoch input hashes are saved.

Native targets are built from the augmented, cropped 512 GT mask and skeleton.
Four native pixels form one work-grid cell. C represents actual native 8-neighbour
cross-cell road adjacency. Disconnected fragments inside a single cell invalidate
its edges. D uses branch tracing up to eight pixels, a three-pixel endpoint/junction
exclusion radius and a 0.7 minimum double-angle resultant. These are initial fixed
settings. GT has no layered crossing information; joined 2D roads are treated as
connected and visually ambiguous annotation cannot be reliably detected.

The native model retains S2 and reads S_H once from Stage3 H before relation
aggregation. One Stage2 global update and one Stage3 neighbour feature update feed
the original surface fusion. Stage3 C/D effective loss multipliers are inherited;
S weights are 0.008/0.012. Per-run effective coefficients are written in
`effective_structure_loss_weights.json`.

The paired validation evaluator uses 14 val images, 512 windows / 256 stride,
weighted logit stitching, no TTA, and thresholds selected by val F1. It freezes
reference gaps once from A's selected checkpoint at 0.45. The records are proxies,
not manually verified true breaks. Fixed-geometry C/D and choice diagnostics
ignore ambiguous labels. It also executes fixed-checkpoint zero-global and
zero-local interventions; they are diagnostic, not deployable model scores.

Before an 80-epoch launch, use `prepare` with the actual data and GPU. CPU synthetic
tests cannot establish actual GPU memory use, data quality, A/B model performance,
or full-run reproducibility. The two old `test_topology_direction` /
`test_final_topology_repair` suites already fail on unmodified 599a410.
