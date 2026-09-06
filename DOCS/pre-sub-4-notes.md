# pre-sub-4 notes (2026-09-05)

## 1. Public-notebook comparison (all NVARC family, same TTT core)

Training hyperparams are byte-identical everywhere: batch 1, accum 1, 1 epoch,
lr 5e-5, cosine, warmup 0.1, aug 16 train / 2 eval, seq 8192, LoRA r64 on
q/k/v/o + gate/up/down + embed + lm_head.

| Notebook (public score) | Precision | Extra machinery | Verdict |
|---|---|---|---|
| Failed in AIMO (33.89) | bf16 | none — plain NVARC baseline | highest score, no architectural secret |
| Learned from AIMO (32.22) | bf16 | `arc_aif.py` object-count beam penalty | penalty has no evidence of helping |
| LB33.89 Minimal Perfpatch (32.22) | bf16 | CPU-transfer speed patch only | speed, not accuracy |
| Ours (pre-sub-4) | 4-bit local / **bf16 on Kaggle** (`ARC_LOAD_4BIT`) | symbolic pre-pass, phase-2 deep pass, diverse attempts, resume, guards | superset of all three |

Takeaway: the only accuracy-relevant edge the public notebooks hold is **full-precision
weights**. Our code already contains everything they do, plus more. The 33.89 top
public score is the plain baseline in bf16 — there is no hidden trick to port.

## 2. Local 8-task eval test (2026-09-05, RTX 4070 8GB, 4-bit, decode_batch 2)

Keys: 0934a4d8, 36a08778, 981571dc, aa4ec2a5 (debug) + e8686506, 28a6681f,
7b5033c1, 20270e3b (cheapest non-debug). 129 shards / 162 samples.

| Task | Train loss | Time | Result |
|---|---|---|---|
| 0934a4d8 | 0.0039 (early-stop @80) | 529s | miss, shards OK |
| 20270e3b | 0.032 | 219s | miss, shards OK |
| 28a6681f | 0.023 (early-stop @96) | 220s | miss, shards OK |
| 36a08778 | 0.0057 | 1279s | half (1/2 outputs) |
| 7b5033c1 | 0.060 | 154s | miss, shards OK |
| 981571dc | 4.7e-05 (early-stop @16) | 769s | solved |
| aa4ec2a5 | 0.0061, train stalled 45x (env noise) → decode timeout | 5851s | no shards → fallback |
| e8686506 | 0.035 (early-stop @80) | 85s | solved |

**Score: 2.5/8 (31%)** — debug-4 subset reproduces sub3's 2.5/4 exactly.
No FATAL guard trips; early-stop callback fired on 4/8 tasks.

## 3. Miss diagnosis: generation, not ranking

For every missed output we checked whether ground truth appears anywhere in its
shards ( Right sheet of paper test):

- 0934a4d8_0, 36a08778_1, 28a6681f_0, 7b5033c1_0, 20270e3b_0, 20270e3b_1:
  truth **never generated** → SEARCH failure (need more views / lower min_prob /
  better training — or bf16 weights).
- aa4ec2a5_0: no shards at all → GENERATION failure (stalled training slot).
- Zero missed outputs had truth present-but-unpicked → no RANKING failures.

So the misses are a search problem, not a selection problem. Scorer tweaks
would not have recovered any of these points.

## 4. Score outlook (honest calibration)

- This pipeline's ceiling is the **32–33 band** (reference 32.22, best public
  33.89). bf16 on Kaggle should put us at/above the reference, toward ranks 9–25.
- **37+ needs a better base model or frontier-scale inference** (see Kaggle
  board: 37.22 / 40.83, then 72.08 / 73.33 — different leagues, e.g. frontier
  multi-channel search + judging at $20–39/task, outside a 4B LoRA budget).
- Floor protection (bronze minimum 31.81): the pre-sub-4 guards make a silent 0%
  repeat impossible — missing Unsloth aborts loudly, zero-loss tasks are skipped,
  systematic zero-loss parks the worker.

## 5. Guards added (pre-sub-4)

- Guard1: `[arc_solver] HAS_UNSLOTH=` banner; RuntimeError on Kaggle without
  Unsloth (`ARC_REQUIRE_UNSLOTH`, default 1 on /kaggle, 0 local). Fallback import
  crashes (e.g. torchao/ScalingType mismatch) are converted to the same message.
- Guard2: per-task `training_loss<=0`/NaN → skip decode loudly; 3 consecutive →
  park worker.
- Guard3: pre-train probe prints first-sequence unmasked-label count (~100+).
- `ARC_LOAD_4BIT`: 1 local, 0 on Kaggle. `ARC_DECODE_BATCH` default 2.

## 6. Fail-fast gates (notebook stops instead of burning quota)

"Save version & Run" now aborts within minutes on any repeat of the 0% failure,
with a message naming the cause:

- Env cell: no unsloth under `/kaggle/usr/lib` or `/kaggle/input` → SystemExit
  (attach the utility-script output first).
- Phase-1 cell: starter `rc != 0` → abort; 0 neural shards → abort.
- Phase-2 cells: `rc != 0` → abort per phase.
- Phase-3 cell: 0 neural shard files across both output dirs → abort instead of
  writing an all-fallback submission.
- `starter.py`: worker-spawn exceptions now `sys.exit(1)` (was: print and return
  0, which let later phases run vacuously).

Limit: only absolute-zero failures trip the gates; partial failures (half the
tasks OOM) still run to assembly.

## 7. Budget allocation controls (more tasks through TTT)

Per-task calibration on RTX 4070 (4-bit): ~0.006–0.018 s/work-unit. The tail
(giants + stalls) eats the fixed budget, so:

- `ARC_TRAIN_CAP=600` (s): `TimeLimitCallback` stops TTT per task after 10 min
  of training. Bounds the aa4ec2a5 stall class (97 min → ≤10 min).
- Adaptive augments: tasks with work > `ARC_WORK_CUT=50000` (~p90) train on
  `ARC_N_TRAIN_AUG_HEAVY=8` augments instead of 16 — 21/240 test tasks.
  Estimator is a local copy of `starter.estimated_work` (verified byte-equal;
  kept local because starter imports this module).
- Composes with early-stop (fires first on easy tasks) and decode `task_cap`.

Expected: heavy-tail train cost roughly halved → on the order of +10–20 extra
tasks through TTT on a 12h 4-GPU rerun. Quality risk bounded to top-decile tasks.

## 8. Repo hygiene

- `.gitignore`: models, outputs, reference downloads, raw sub1/2/3 archives stay
  untracked. `pre-sub-4/`, `DOCS/`, sources, utility fork stay committable.
- Branches: `main` (all work) + `original` (2-commit pre-work state), both pushed.
- `utility/`: own pinned unsloth fork (torch 2.8.0, cp312 flash-attn, qwen3 patch
  applied, output snapshot verified) — attach its output in the Kaggle notebook.
