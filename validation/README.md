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
