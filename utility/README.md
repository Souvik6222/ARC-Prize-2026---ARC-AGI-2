# Own Unsloth builder (fork) — status & plan

## Goal
Publish our own pinned copy of `pip-install-unsloth-flash-patch` under our account so the
ARC notebook never depends on someone else's utility script being attached/available.
The notebook's saved **output** (`unsloth/`, `torch`, `flash_attn` under `/kaggle/working`)
is what gets mounted offline at `/kaggle/usr/lib/notebooks/.../` in the ARC run
(proven healthy in `submission/sub3`, which scored 2.5/4 on debug keys).

## File
- `pip-install-unsloth-flash-patch(fork).ipynb` — fixed locally, ready for Kaggle
  File → Import Notebook. **Rename the Kaggle title to
  `pip-install-unsloth-flash-patch(fork)` BEFORE the first save** (slug freezes on save).

## Fix 1 — cell 0: pin `torch==2.8.0`
Was unpinned (`torch>=2.4.0` via unsloth deps) → pip resolved **torch 2.14.0 + CUDA 13
tree**, untested with Unsloth 2025.9.7 and wrong ABI for the flash-attn wheel.
Now:
```
!pip install --target=/kaggle/working torch==2.8.0 unsloth==2025.9.7 unsloth_zoo==2025.9.9 numpy==2.2.6 matplotlib==3.10.6 scikit-learn==1.7.2
```
(torch 2.8.0 = the proven sub3 combo with transformers 4.55.4.)

## Fix 2 — cell 1: `cp311` → `cp312` wheel
Kaggle's current image is **Python 3.12.13**; the old `...cu128torch2.8-cp311-cp311...`
wheel errors with "not a supported wheel on this platform". The `cp312` asset was
verified to exist in `v0.3.14`. Now:
```
!pip install --no-deps --no-build-isolation --target=/kaggle/working https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp312-cp312-linux_x86_64.whl
```

## Evidence (failed run: `pip-install-unsloth-flash-patch-fork_1.log`)
- `Successfully installed ... torch-2.14.0 ...` (pin missing)
- `ERROR: flash_attn-2.8.2+cu128torch2.8-cp311-cp311-linux_x86_64.whl is not a supported wheel`
- `ModuleNotFoundError: No module named 'flash_attn'` → version saved with no usable output
- (The long red dependency-conflict block in that log is harmless noise: pip complaining
  about system packages; `--target=/kaggle/working` is isolated.)

## Kaggle run steps (one more try)
1. Delete the broken draft, Import Notebook with the fixed local file.
2. Session options: Internet **ON**. Accelerator **None/CPU if offered** — this is a pure
   pip-install run; if only GPU/TPU is offered it costs ~10 min of GPU quota.
3. Run All → expect `torch 2.8.0`, working `import flash_attn`, patch applies cleanly.
4. Save Version → **Save & Run All** (NOT Quick Save — no output otherwise).
5. Verify the Output tab contains `unsloth/` (+ `qwen3.patch`).
6. Attach in ARC notebook via + Add Input (Your Work + Utility Scripts); confirm the log
   shows the fork path in the `utility scripts:` line and the Unsloth banner per rank.

## Plan B (fallback, zero Kaggle compute) — dataset upload
If the fork fails again, skip Kaggle execution entirely:
1. Locally: `pip download` the exact pinned wheels (torch 2.8.0, unsloth 2025.9.7, deps,
   cp312 flash-attn), unpack into a folder with `unsloth/` at top level, apply the qwen3
   patch locally.
2. Upload the folder as a **private Kaggle dataset** from the browser (no run, no quota).
3. Attach the dataset to the ARC notebook. Discovery needs **no change**:
   `my_notebook/arc_solver.py:34` already globs `/kaggle/input/**/unsloth`.
4. Independent of A/B: pre-sub-4 still needs the sys.path priority flip so the
   self-consistent prebuilt stack (torch 2.8.0) wins over the system torch.

## Status (2026-09-05): PUBLISHED, output verified complete
Third run (fixed file) succeeded after a frozen-looking finalize phase — the Output
snapshot committed fine (~6 min total compute). Output contains the full consistent stack:
`torch` + `torch-2.8.0.dist-info`, `transformers-4.55.4`, `triton-3.4.0`,
`xformers-0.0.32.post2`, `trl-0.22.2`, `torchao-0.18.0`, `unsloth` + `unsloth-2025.9.7.dist-info`,
`unsloth_zoo` + `unsloth_zoo-2025.9.9`, `flash_attn_2_cuda` .so, `qwen3.patch`.
Log proved: `Successfully installed flash-attn-2.8.2`, patch applied to
`unsloth/models/qwen3.py`, no resolver errors. Plan B (dataset route) stood down.
