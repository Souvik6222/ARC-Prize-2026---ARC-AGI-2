# ARC-AGI-2 Error & Troubleshooting Log Base

> **Systematic incident tracker, root cause analyses, failure modes, and debugging protocols for the Kaggle ARC Prize 2026 workspace.**

---

## 1. Incident Logging Protocol

Whenever an error, crash, OOM, timeout, or submission anomaly occurs, record it using this standard schema:

```markdown
### [ERR-XXX] Short Title describing the failure
* **Date & Run Ref**: YYYY-MM-DD | Notebook / Script: `try__X/...`
* **Severity**: [Critical (Fatal crash / zero score) | High (Task failure) | Medium (Suboptimal fallback)]
* **Environment**: (e.g. 4x L4 GPUs, Kaggle Linux, PyTorch 2.4, CUDA 12.2)
* **Symptom / Traceback**:
  ```
  <Paste traceback or error snippet here>
  ```
* **Root Cause**: Deep explanation of why the failure occurred.
* **Resolution / Workaround**: The exact fix or defensive code added.
* **Verification**: How it was tested and confirmed fixed.
* **Prevention Rule**: Rule/assertion to enforce going forward.
```

---

## 2. Historical & Known Failure Modes Catalog

### [ERR-001] TensorFlow / Triton PTX Compiler Conflict
* **Severity**: `Critical`
* **Component**: Kernel JIT / Unsloth initialization
* **Symptom**:
  ```
  RuntimeError: Triton compiler failed to find ptxas or shared library conflict with libdevice
  ```
* **Root Cause**: Kaggle environment pre-installs TensorFlow which registers conflicting CUDA runtime path references and bundled `ptxas` binaries incompatible with Triton's JIT.
* **Resolution**:
  Uninstall TensorFlow in the first notebook cell before any PyTorch or Triton imports:
  ```python
  import subprocess, sys
  subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "tensorflow"], check=False)
  os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
  ```
* **Prevention Rule**: Always include the uninstall command as the top execution block in all GPU notebooks.

---

### [ERR-002] Multi-Task Test-Time Training CUDA Memory Accumulation (OOM)
* **Severity**: `Critical`
* **Component**: `arc_solver.py` / LoRA Fine-Tuning Loop
* **Symptom**:
  ```
  torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB (GPU 0; 22.19 GiB total capacity; 21.80 GiB already allocated)
  ```
* **Root Cause**: Initializing and deleting Hugging Face / Unsloth LoRA adapters in a continuous loop leaves PyTorch caching allocator fragmented and gradient state uncollected across iterations.
* **Resolution**:
  Explicitly destroy references and flush both Python garbage collection and CUDA cache after solving each individual task:
  ```python
  del trainer, model, lora_model
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.ipc_collect()
  ```
* **Prevention Rule**: Wrap every task solve call inside a `try...finally` block that guarantees memory reclamation even if an exception occurs during decoding.

---

### [ERR-003] Worker Process Deadlock in Multiprocessing Queue
* **Severity**: `High`
* **Component**: `starter.py` / Multi-GPU Worker Pool
* **Symptom**:
  Notebook hangs indefinitely when 1 GPU worker encounters an unhandled exception or task timeout.
* **Root Cause**: The parent process waits on a `mp.Queue` for task completion, but if a worker dies abruptly or hangs on DFS recursion without emitting a sentinel, the queue never resolves.
* **Resolution**:
  1. Implement a per-task watchdog timeout (`task_cap` parameter, e.g. 1200 seconds).
  2. Use explicit `None` sentinels to signal worker completion.
  3. Wrap the worker inner loop in an exception catcher that logs the traceback to `/logs/worker_X.log` and pushes an empty/error result to the queue rather than crashing the worker.
* **Prevention Rule**: Never allow a worker subprocess to terminate silently without writing to the shared communication queue.

---

### [ERR-004] Malformed Submission Schema & Shape Violations
* **Severity**: `Critical`
* **Component**: `submission.json` generation & serialization
* **Symptom**:
  Kaggle submission evaluation returns `Submission Scoring Error` or zero score.
* **Root Cause**:
  * Generated grid contains floating point numbers instead of integers (`1.0` vs `1`).
  * Grid colors outside $[0, 9]$ (e.g. $-1$, $10$, or NaN).
  * Grid dimensions exceed $30 \times 30$ or contain empty rows `[[]]`.
  * Missing keys (`attempt_1` or `attempt_2`).
* **Resolution**:
  Enforce a strict post-generation schema validator before writing `submission.json`:
  ```python
  def validate_submission_schema(submission, test_challenges):
      assert set(submission.keys()) == set(test_challenges.keys()), "Task key mismatch"
      for tid, attempts in submission.items():
          assert isinstance(attempts, list) and len(attempts) == len(test_challenges[tid]["test"])
          for pair in attempts:
              for key in ["attempt_1", "attempt_2"]:
                  assert key in pair, f"Missing {key}"
                  grid = pair[key]
                  assert isinstance(grid, list) and len(grid) > 0 and len(grid) <= 30
                  w = len(grid[0])
                  assert w > 0 and w <= 30
                  for row in grid:
                      assert len(row) == w, "Ragged row lengths"
                      for val in row:
                          assert isinstance(val, int) and 0 <= val <= 9, f"Invalid cell: {val}"
  ```
* **Prevention Rule**: Never save `submission.json` without running `validate_submission_schema()`.

---

### [ERR-005] 12-Hour Kaggle Rerun Wall-Clock Expiration
* **Severity**: `Critical`
* **Component**: Global Scheduler / Time Budgeting
* **Symptom**:
  Notebook killed by Kaggle at $12:00:00$ mark before writing final `submission.json`.
* **Root Cause**: Fixed number of iterations per task multiplied by unexpected long decoding runs on complex tasks exceeds the 12-hour limit.
* **Resolution**:
  * Set hard cutoff at `12 * 3600 - 600` (10-minute reserve).
  * Process tasks ordered cheap-first (smallest token footprint first).
  * Check `time.time() > global_end_time` before starting each task and before entering Phase 2 deep search.
  * Persist intermediate candidate files to disk atomically after each task so progress is never lost.
* **Prevention Rule**: Always guarantee atomic disk writes after each task completion.

---

## 3. Active Incident Log & Debugging Tracker

| Incident ID | Date | Component | Error Description | Status | Resolved By |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ERR-001` | 2026-09-02 | Unsloth / Triton | TensorFlow PTX Library Conflict | `RESOLVED` | Uninstall TF in preflight |
| `ERR-002` | 2026-09-02 | LoRA TTT | CUDA Allocator fragmentation across tasks | `RESOLVED` | `gc.collect()` + `empty_cache()` |
| `ERR-003` | 2026-09-02 | Multiprocessing | Worker deadlock on unhandled task error | `RESOLVED` | Sentinel queue + per-task timeout |
| `ERR-004` | 2026-09-02 | Output Serializer | Risk of Kaggle submission scoring error | `RESOLVED` | Fail-closed schema validator |
| `ERR-005` | 2026-09-02 | Runtime Budget | 12h Kaggle timeout risk | `RESOLVED` | Cheap-first + 10m buffer check |

*(New incidents will be appended below)*

### [ERR-006] arc_symbolic.py Failure - 2026-09-02 10:17:03
* **Date & Run Ref**: 2026-09-02 10:17:03 | Component: `arc_symbolic.py`
* **Severity**: `Medium`
* **Symptom / Traceback**:
```
Task non-invertible grid shape
```
* **Root Cause**: Unequal train/test dimensions in rare ARC task
* **Resolution / Mitigation**: Handled safely with try-except returning empty symbolic candidate list
* **Status**: `LOGGED`

---

### [ERR-007] Notebook Cell 1 Failure - 2026-09-02 10:42:50
* **Date & Run Ref**: 2026-09-02 10:42:50 | Component: `Notebook Cell 1`
* **Severity**: `Low`
* **Symptom / Traceback**:
```
NameError: name 'ARC_MODEL_PATH' is not defined in print f-string
```
* **Root Cause**: Missing single quotes around dictionary key inside os.environ['ARC_MODEL_PATH']
* **Resolution / Mitigation**: Fixed to os.environ.get('ARC_MODEL_PATH')
* **Status**: `LOGGED`

---

### [ERR-008] arc_loader.py & Cell 11 Failure - 2026-09-02 10:47:22
* **Date & Run Ref**: 2026-09-02 10:47:22 | Component: `arc_loader.py & Cell 11`
* **Severity**: `Medium`
* **Symptom / Traceback**:
```
AttributeError: 'str' object has no attribute 'keys' when passing filepath string to ArcDataset()
```
* **Root Cause**: ArcDataset constructor expected dictionary queries, whereas ArcDataset.from_file(test_path) loads from JSON file
* **Resolution / Mitigation**: Updated ArcDataset constructor to auto-load JSON when given a string path, and updated Cell 10/11 to use ArcDataset.from_file()
* **Status**: `LOGGED`

---

### [ERR-009] Notebook Cell 11 Failure - 2026-09-02 10:49:31
* **Date & Run Ref**: 2026-09-02 10:49:31 | Component: `Notebook Cell 11`
* **Severity**: `Low`
* **Symptom / Traceback**:
```
NameError: name 'rb' is not defined in open(submission_file, rb)
```
* **Root Cause**: Missing quotation marks around file mode string 'rb' inside f-string interpolation
* **Resolution / Mitigation**: Extracted file hash calculation into explicit with open(submission_file, 'rb') block
* **Status**: `LOGGED`

---

### [ERR-010] arc_solver.py Failure - 2026-09-02 10:53:30
* **Date & Run Ref**: 2026-09-02 10:53:30 | Component: `arc_solver.py`
* **Severity**: `High`
* **Symptom / Traceback**:
```
ModuleNotFoundError: No module named 'unsloth' in worker subprocess
```
* **Root Cause**: Unsloth library not pre-installed in default Kaggle environment
* **Resolution / Mitigation**: Added automated Unsloth discovery from /kaggle/usr/lib and offline wheels, plus a pure HuggingFace transformers+peft fallback in arc_solver.py
* **Status**: `LOGGED`

---

### [ERR-011] starter.py Failure - 2026-09-02 11:05:32
* **Date & Run Ref**: 2026-09-02 11:05:32 | Component: `starter.py`
* **Severity**: `High`
* **Symptom / Traceback**:
```
RuntimeError: A SemLock created in a fork context is being shared with a process in a spawn context
```
* **Root Cause**: Using raw mp.Queue() when PyTorch multiprocessing spawn is used on Python 3.12
* **Resolution / Mitigation**: Switched to mp.Manager().Queue() which is safely proxyable across both fork and spawn contexts
* **Status**: `LOGGED`

---

### [ERR-012] arc_solver.py / Transformers Failure - 2026-09-02 11:20:28
* **Date & Run Ref**: 2026-09-02 11:20:28 | Component: `arc_solver.py / Transformers`
* **Severity**: `High`
* **Symptom / Traceback**:
```
ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported
```
* **Root Cause**: Outdated torchao 0.10.0 installed in Kaggle base image causes newer transformers weight materialization to abort
* **Resolution / Mitigation**: Uninstalled torchao in Cell 2 preflight and set sys.modules['torchao'] = None in arc_solver.py
* **Status**: `LOGGED`

---

### [ERR-013] arc_solver.py / UnslothFixedTrainer Failure - 2026-09-02 11:40:09
* **Date & Run Ref**: 2026-09-02 11:40:09 | Component: `arc_solver.py / UnslothFixedTrainer`
* **Severity**: `High`
* **Symptom / Traceback**:
```
TypeError: Trainer.__init__() got an unexpected keyword argument 'tokenizer'
```
* **Root Cause**: Newer HuggingFace Transformers (>=4.46) renamed 'tokenizer' to 'processing_class' and removed Unsloth-specific kwargs in Trainer.__init__
* **Resolution / Mitigation**: Updated UnslothFixedTrainer.__init__ to dynamically inspect Trainer signature and convert tokenizer to processing_class and strip unsloth kwargs when in fallback mode
* **Status**: `LOGGED`

---

### [ERR-014] arc_solver.py / QwenDataCollatorForCompletionOnlyLM Failure - 2026-09-02 11:50:05
* **Date & Run Ref**: 2026-09-02 11:50:05 | Component: `arc_solver.py / QwenDataCollatorForCompletionOnlyLM`
* **Severity**: `High`
* **Symptom / Traceback**:
```
NameError: name 'QwenDataCollatorForCompletionOnlyLM' is not defined
```
* **Root Cause**: During previous Trainer replacement, the custom collator and turbo_dfs functions were omitted from the cell payload
* **Resolution / Mitigation**: Reconstructed full arc_solver.py including QwenDataCollatorForCompletionOnlyLM, turbo_dfs, and calc_scores
* **Status**: `LOGGED`

---

### [ERR-015] arc_solver.py / make_training_args Failure - 2026-09-02 11:56:07
* **Date & Run Ref**: 2026-09-02 11:56:07 | Component: `arc_solver.py / make_training_args`
* **Severity**: `High`
* **Symptom / Traceback**:
```
NameError: name 'make_training_args' is not defined
```
* **Root Cause**: Function make_training_args was missing from arc_solver.py file definition
* **Resolution / Mitigation**: Reconstructed full arc_solver.py with make_training_args, UnslothFixedTrainer, and all required helper functions
* **Status**: `LOGGED`

---
