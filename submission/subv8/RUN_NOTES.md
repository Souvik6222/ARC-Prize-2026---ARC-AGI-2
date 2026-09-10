# Sub-8 push + check run (2026-09-10/11) — `soukeaizenz/arc2-hybrid-v8` v1

## What was pushed
- Notebook: `submission/presubv8/arc2-hybrid-v8.ipynb` (fast path, shield +
  index fix, Cell-12 shield, r128, bf16-on-Kaggle, all guards, budget cap unset)
- Pushed via CLI (not browser): `kaggle kernels push` with `kernel-metadata.json`
  (GPU on, internet off, machine L4, inputs: competition data + sorokin model +
  own fork output). New kernel `arc2-hybrid-v8`, version 1.

## Check run result: GREEN in ~6 min (fast path, by design)
- `rerun=False`, `FAST_PATH=True (ARC_FULL_RUN=0)`, budget line full (no cap)
- Env gate: fork bundle found; flip applied (`bundle roots at sys.path front`)
- Symbolic prepass: 120 eval tasks in 0.70s, 0 exact (matches local measurement)
- Phase-2 auto-skipped; phase-3 wrote **240 tasks / 259 outputs, schema OK**,
  sha256 `3739be35dbba` (this folder's `submission.json`, hash-verified match)
- `arc2-hybrid-v8_run1.log` in this folder is the full 52-line log

## Known gap (accepted residual risk)
Fast path never spawns workers, so `from unsloth import ...` is unexecuted
here. First real import happens in the scoring rerun; the broad guard turns any
failure into a loud abort, not a silent 0%.

## Submitted
Version 1 submitted to ARC Prize 2026 - ARC-AGI-2 (1 submission/day limit —
this was the day's shot). Scoring rerun: full 240-task TTT expected ~6–10h on
competition compute (11.83h limit, own quota untouched).
