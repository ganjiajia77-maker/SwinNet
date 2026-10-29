# Executed checks

- `D:\torch-cu128\Scripts\python.exe checks/test_v9_multiscale.py -v`
  passed 8 tests. Covers exact local and blend=0 equivalence, RNG isolation,
  fractional coordinate convention, source validity, stay fallback, convexity,
  finite gradients/frozen provider, strict saves/loads, fixed-gap closure versus
  a disconnected center island, training/resume and the original evaluator's
  four-value return interface.
- `checks/smoke_v9_multiscale_cuda.py` passed batch 1 and batch 2, 512 input,
  128 work grid, two optimizer steps each, original and local logit error 0.
  Details and actual CUDA memory/time are in `multiscale_cuda.json`.
- `checks/smoke_v9_multiscale_native.py` passed original/local/multiscale strict
  downstream reloads, native1500 weighted512/stride256 inference (25 windows per
  arm), original confusion-count reproduction and all diagnostic output files.
  See `multiscale_native.json`.
- Python compilation and `bash -n scripts/run_v9_multiscale.sh` passed.

All runtime checks here use a synthetic frozen feature provider. Full BASE/Q3
weights and Massachusetts images are on the server and were not available in
these checks. They therefore do not establish real-data accuracy, real-backbone
peak VRAM or exact reproduction of the user's trained checkpoint. The server
launcher requires its separate real-checkpoint/real-crop smoke before training.

No formal training was started.

## Allocator compatibility follow-up

The original server failure was at a CUDA allocation during source scoring.
We removed the forced expandable-segment allocator and reduced preflight live
tensors. No network/scoring/dynamics/loss formulas were changed.

The updated `checks/test_v9_multiscale.py` contains 9 checks and was additionally
run with PyTorch 2.6.0+cpu (local `.codex_work/v9-cpu-deps`). The controller-only
test stubs its external evaluation imports, avoiding an unrelated mismatch
between this CPU torch and the host's newer compiled torchvision.

`multiscale_cuda_allocator_fix.json` records a fresh CUDA run with the native
allocator, both 512 batch sizes and the actual updated entry-point smoke. This
host uses PyTorch2.11/cu128, not the server's PyTorch2.6/cu118/Quadro driver.
The precise server-side driver failure therefore remains unconfirmed until its
real smoke rerun. New smoke runs synchronize CUDA and include C++ stacks if an
error recurs; a failed smoke always stops the formal pipeline.

## Occupied GPU follow-up

`checks/test_v9_gpu_preflight.py` passes three stdlib-only tests: physical-index
to UUID selection and same-card process filtering; an occupied-card stop without
killing any process; rejection of missing/ambiguous GPU responses. The shell's
syntax is checked with bash -n. No model, loss, data or training-runner file was
changed in this follow-up, preserving existing checkpoint code fingerprints.
The 6144MiB reserve check is an early occupancy guard, not an exclusive GPU lock
or a guarantee of sufficient peak VRAM. Actual server occupancy must be resolved
before continuing the interrupted experiment.

`checks/test_v9_continue_shell.py` passes two additional real-Bash controller
tests with a fake Python: it resumes local weights, preserves an incomplete
multiscale directory, and executes all nine pipeline phases in order; a busy
GPU exits before invoking model code or moving any interrupted output directory.
These tests never start training or query/terminate a real GPU process.
