# presubv8 snapshot (2026-09-10) — fast path + merged shield + r128

Submit notebook (source of truth):
`/home/arch/ipynb/ARC-Prize-2026---ARC-AGI-2/my_notebook/arc2-hybrid-v3.ipynb`
(this folder is the frozen checkpoint of the same content).

## What changed vs pre-sub-4

1. **Fast path (v7-style).** Interactive runs (`RERUN` unset) skip GPU TTT:
   symbolic pre-pass only over eval tasks (~0.25s for 120 tasks), then baseline
   assembly — minutes and ~zero quota, like v7's 3m43s check run. Escape hatch
   `ARC_FULL_RUN=1` forces the 4-task TTT proof run. Scoring reruns always do
   full 240-task TTT. Zero-shard fail-fast gates are bypassed in fast mode (a
   symbolic+identity baseline is the point there) but stay armed in RERUN.
2. **Shield ported to our line** (`arc_decoder.py`): `_fb_test_input()` handles
   split AND non-split dataset layouts (naive `test[0]` wrong on one, naive
   `test[i]` crashes on the other); patched `select_orthogonal_top2` falls back
   to the test input instead of `[[0]]`; same fix applied to
   `run_selection_algo` (it fed `score_v2` wrong inputs too). Verified: toy
   cases + real wiring on multi-test task `12997ef3_1`, both layouts pass.
3. **Cell-12 shield** (builder phase-3): dummy/duplicate `attempt_2` becomes the
   test input unless `attempt_1` already is it. Authoritative layer — immune to
   every indexing bug class.
4. **r=128 Kaggle default** (64 local), `ARC_LORA_R` override; `use_rslora`
   was already on. Measured: +14%/step vs r64 (+33% for r256), 4.5GB peak,
   best loss of the three on the probe task (single-task noise — precedent,
   not proof).
5. Carried over: `ARC_LOAD_4BIT` (bf16 Kaggle / 4-bit local), train-cap 600s,
   adaptive 8-aug for work>50000, bundle flip, broad Unsloth guard, loss/label
   guards, `ARC_BUDGET_HOURS` quota cap (unset by default — full 11.83h scoring).

## Floor math (honest version)

202 identity + 7 symbolic = 209/720 = 29.03% locked by shields (the "15
symbolic pairs" claim is irreproducible: the 7 tasks hold 7 pairs; the 8 extra
pairs in the 30.14% story are unexplained — most plausibly sub-5 neural
solves). +~15 neural-only solves observed in sub-7 → ~32% if neural holds.

## Pre-push checklist

- [ ] Notebook imported from `my_notebook/arc2-hybrid-v3.ipynb` (NOT an old copy)
- [ ] Inputs: competition data + `sorokin/qwen3_4b_grids15_sft139` + own fork output
- [ ] GPU L4 ×4, Internet OFF, Save & Run All (not Quick Save)
- [ ] Log line 1: `rerun=False`, `FAST_PATH=True`, `[budget]` full (no cap)
- [ ] `unsloth candidates:` non-empty → `bundle roots at sys.path front:`
- [ ] No `ARC_BUDGET_HOURS` set anywhere (would truncate the scoring rerun)
- [ ] After green: Submit to competition (scoring runs on competition compute)
