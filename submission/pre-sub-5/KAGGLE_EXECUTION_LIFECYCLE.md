# Kaggle GPU Quota Accounting & Competition Execution Lifecycle

**Date:** 2026-09-07  
**Scope:** ARC Prize 2026 / ARC-AGI-2 Code Competition Architecture  
**Author:** AI Engineering & Research Team

---

## 1. Executive Summary

In Kaggle Code Competitions, notebook execution is bifurcated into two distinct phases with entirely separate infrastructure, execution environments, and GPU quota accounting rules:

1. **Commit Phase ("Save & Run All" / `kaggle kernels push`)**:
   - Runs in the user's interactive/commit sandbox.
   - **Billed against the user's personal weekly GPU quota (30 hours/week limit).**
   - The competition rerun flag `KAGGLE_IS_COMPETITION_RERUN` is **NOT set** (`False`).
   - Its sole purpose is to produce a valid `submission.json` output file so the kernel version can be selected for leaderboard submission.
   - **Target Duration:** **~25–35 minutes** (running a small sanity subset of 4 debug tasks across all 4 GPUs).

2. **Competition Scoring Phase ("Submit to Competition" / Auto-Grading Rerun)**:
   - Triggered automatically by Kaggle when a finished kernel version's `submission.json` is submitted to the leaderboard.
   - **Billed to dedicated Kaggle competition compute (0 hours deducted from user's personal quota).**
   - Kaggle sets `KAGGLE_IS_COMPETITION_RERUN=1`.
   - Kaggle dynamically swaps the dummy/public test directory with the secret hidden test set (100–240 tasks).
   - **Target Duration:** **Up to 11 hours 50 minutes** (full dynamic budget utilization on 4× L4 GPUs).

---

## 2. Detailed Accounting & Infrastructure Comparison

| Dimension | Commit Mode ("Save & Run All" / `kernels push`) | Competition Scoring Rerun Mode |
|---|---|---|
| **Trigger** | User clicking "Save & Run All" or `kaggle kernels push` | User clicking "Submit to Competition" or `kaggle competitions submit` |
| **GPU Quota Charged** | **Personal User Quota** (30 hours/week max) | **Dedicated Competition Compute** (**Free / 0 personal quota**) |
| **Environment Variable** | `KAGGLE_IS_COMPETITION_RERUN` is unset (`None` / empty) | `os.environ["KAGGLE_IS_COMPETITION_RERUN"] == "1"` |
| **Dataset Exposed** | Public dummy test or evaluation challenges (~100 tasks) | Hidden test challenges (private test set, ~100–240 tasks) |
| **Expected Budget** | **35–55 minutes** (short debug pass) | **11 hours 50 minutes** (max allowed competition window) |
| **Tasks Evaluated** | 4 debug tasks (`0934a4d8, 36a08778, 981571dc, aa4ec2a5`) | All hidden test tasks (100–240 tasks sorted cheap-first) |
| **Primary Objective** | Fast sanity check: verify imports, CUDA allocation, output schema | Maximum competitive score: solve as many test tasks as possible |

---

## 3. Forensics: Why Sub-3 vs Sub-4 / Sub-5 Behaved Differently

### Submission 3 Behavior (~26–30 min Commit, 12h Scoring)
In Submission 3 (`arc2-nvarc-v2_fork.ipynb`), the code correctly implemented the **dual-mode gate**:

```python
# In Notebook Cell 1:
RERUN = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
T0 = time.time()
if RERUN:
    global_end_time = T0 + 12 * 3600 - 600  # 11h 50m for hidden competition test
else:
    global_end_time = T0 + 55 * 60          # 55 min for manual commit

# In starter.py:
if not rerun_mode:
    debug_keys = os.getenv("ARC_DEBUG_KEYS", "0934a4d8,36a08778,981571dc,aa4ec2a5").split(",")
    keys = [k for k in keys if k in debug_keys]
```

- **During Commit:** `rerun_mode` was `False`. The worker processed only 4 tasks on 4 GPUs. It completed in ~26 minutes, verified all pipeline components, consumed only **0.43 hours** of the user's weekly 30h quota, and wrote `/kaggle/working/submission.json`.
- **During Scoring:** When submitted to the leaderboard, Kaggle launched the hidden test rerun with `KAGGLE_IS_COMPETITION_RERUN=1`. The script processed all tasks for 11h 50m on Kaggle's dedicated competition compute without touching the user's personal quota.

---

### The Sub-4 / Early Sub-5 Misconception
In Sub-4 and the initial Sub-5 setup, a code comment revealed a critical misunderstanding of Kaggle's architecture:

```python
# INCORRECT REASONING FOUND IN STARTER.PY:
# "NOTE: debug key filter removed — ARC_DEBUG_KEYS only applied when explicitly set.
# KAGGLE_IS_COMPETITION_RERUN is NOT set during user 'Save & Run' submissions,
# only during the official competition rerun (auto-grading). Previously this
# block was filtering ALL manual submissions to only 4 debug tasks, causing
# 32-min runs instead of 12-hour full 240-task runs."
```

#### The Fatal Flaw in that Logic:
1. The author assumed that a "Save & Run All" commit was supposed to be the 12-hour evaluation run.
2. In reality, **manual commits are NOT scored by the competition leaderboard**.
3. Removing the debug filter caused the manual commit to attempt solving all 100+ evaluation tasks for 11.33 hours **on the user's personal 30h weekly quota**.
4. Result: The user had to wait 11+ hours just to obtain a committed notebook version, and burned ~11.5 hours of their weekly GPU quota in the process.

---

## 4. Daily Submission Limits vs Kernel Pushes

A common source of confusion is the relationship between **Kernel Pushes** and **Competition Submissions**:

1. **Kernel Pushes (`kaggle kernels push` / "Save & Run All")**:
   - **Limit:** Unlimited (constrained only by your 30 hours/week personal GPU quota).
   - Pushing a kernel simply runs your notebook on Kaggle hardware and saves the output files (e.g., `submission.json`) in the kernel's output viewer.
   - Pushing a kernel **does NOT count as a competition submission**.

2. **Competition Submissions (`kaggle competitions submit` / "Submit to Competition")**:
   - **Limit:** Strict **1 submission per UTC day** for the ARC-AGI-2 competition.
   - You take the output of a completed kernel version and submit it to the competition leaderboard.
   - Once submitted, Kaggle boots a private worker with `KAGGLE_IS_COMPETITION_RERUN=1` to grade your model on hidden test data.

### Operational Decision for Sub-5 (Version 1):
Because Version 1 of `arc2-hybrid-v5` was already running stably for 2.7+ hours on 4× L4 GPUs without error, and because the user is constrained by the 1 submission/day rule:
- **Decision:** Let Version 1 continue its run to completion.
- Once finished, its output `submission.json` will represent a verified, full-pipeline run with all architectural enhancements (left-truncation, $r=256$ RSLoRA, zero-loss guard, pure consensus `score_kgmon`) and will be submitted as today's 1 competition entry.

---

## 5. Universal Dual-Mode Architecture (Standard for Sub-6+)

To prevent burning personal GPU quotas in future iterations (Sub-6 and beyond), all future submissions **must** enforce the following standard:

### Specification 1: Notebook Header (Cell 1)
```python
# ---------------------------------------------------------------------------
# Global Wall-Clock Budget Gate
# ---------------------------------------------------------------------------
import os, time, json

RERUN = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
global_start_time = time.time()

if RERUN:
    # Dedicated competition compute: run full 11h 50m window (10m safety buffer)
    budget_hours = 11.8333
else:
    # Commit mode: fast sanity pass (~30 min), saves personal weekly GPU quota
    budget_hours = 0.60

global_end_time = global_start_time + budget_hours * 3600
print(f"[BUDGET] rerun_mode={RERUN} | budget={budget_hours:.2f}h | cutoff={time.ctime(global_end_time)}")
```

### Specification 2: Task Orchestrator (`starter.py`)
```python
    rerun_mode = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
    
    if args.keys_file and os.path.exists(args.keys_file):
        with open(args.keys_file) as f:
            keys = [k for k in json.load(f) if k in data]
    else:
        keys = sorted(data.keys())
        if not rerun_mode:
            # Commit mode: process only 4 diverse debug keys to verify end-to-end
            # pipeline health without exhausting personal weekly GPU quota.
            debug_keys_env = os.getenv("ARC_DEBUG_KEYS", "0934a4d8,36a08778,981571dc,aa4ec2a5")
            debug_keys = [k.strip() for k in debug_keys_env.split(",") if k.strip()]
            keys = [k for k in keys if k in debug_keys]
            print(f"[COMMIT-MODE] Filtered {len(data)} tasks -> {len(keys)} debug tasks: {keys}")
        else:
            print(f"[RERUN-MODE] Official competition scoring active! Processing all {len(keys)} test tasks.")
```

---

## 6. Summary Checklist for Future Submissions

- [x] **Verify `KAGGLE_IS_COMPETITION_RERUN` Gate:** Never remove the `not rerun_mode` filter.
- [x] **Commit Time Target:** Manual commits should ALWAYS finish in **25–35 minutes**.
- [x] **Quota Monitoring:** Check personal GPU quota in Kaggle Settings before long interactive debugging.
- [x] **Zero-Loss Guard & Baseline Pre-population:** Always pre-populate `submission.json` at second 0 so any early timeout still registers a valid submission.
- [x] **Daily Submission Discipline:** Only 1 submission per day can be graded on the private leaderboard. Plan kernel commits ahead of the UTC cutoff.
