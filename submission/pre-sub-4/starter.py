"""
starter.py - Multi-GPU Orchestrator with Integrated Fast Symbolic Pre-Pass
"""
import os
# Fix6: must be set before torch CUDA init
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import sys, glob

# Discover unsloth bundle roots (from attached utility scripts or input datasets)
_unsloth_search_paths = (
    glob.glob("/kaggle/usr/lib/notebooks/*/pip_install_unsloth*") +
    glob.glob("/kaggle/usr/lib/notebooks/*/*/pip_install_unsloth*") +
    glob.glob("/kaggle/usr/lib/**", recursive=True) +
    glob.glob("/kaggle/input/**/unsloth", recursive=True)
)
_bundle_roots = []
for p in _unsloth_search_paths:
    if os.path.isdir(p):
        if os.path.isfile(os.path.join(p, "unsloth", "__init__.py")):
            if p not in _bundle_roots:
                _bundle_roots.append(p)
        elif os.path.basename(p) == "unsloth" and os.path.isfile(os.path.join(p, "__init__.py")):
            parent = os.path.dirname(p)
            if parent not in _bundle_roots:
                _bundle_roots.append(parent)

# Prepend bundle roots to sys.path and PYTHONPATH before importing torch so that
# the self-contained torch 2.8.0 + torchvision + transformers 4.55.4 + unsloth 2025.9.7
# stack is imported consistently in starter and all spawned workers.
for _b in reversed(_bundle_roots):
    if _b in sys.path:
        sys.path.remove(_b)
    sys.path.insert(0, _b)
if _bundle_roots:
    _cur = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = ":".join(_bundle_roots + ([_cur] if _cur else []))
    print(f"[starter] bundle roots at sys.path front: {_bundle_roots}", flush=True)

import time
import json
try:
    import torch
    import torch.multiprocessing as mp
except ImportError:
    torch = None
    import multiprocessing as mp

# Compatibility shims for torch 2.8.0 + transformers/torchao
try:
    import torch
    import torch.nn.functional as F
    from enum import Enum

    if not hasattr(F, "ScalingType"):
        class ScalingType(Enum):
            DELAYED = "delayed"
            DYNAMIC = "dynamic"
            Delayed = "delayed"
            Dynamic = "dynamic"
        F.ScalingType = ScalingType

    if not hasattr(F, "scaled_grouped_mm"):
        def _scaled_grouped_mm(*args, **kwargs):
            return None
        F.scaled_grouped_mm = _scaled_grouped_mm
except Exception:
    pass

try:
    import transformers.utils.import_utils as _tiu
    _tiu.is_torchao_available = lambda: False
except Exception:
    pass

import argparse
import traceback
import bz2
import pickle
import numpy as np

from arc_symbolic import solve_task_symbolic
from arc_invariants import is_valid_grid


def local_worker(rank, queue, end_time, test_path, marker_dir):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.set_default_device("cpu")

    # Stagger worker imports to prevent simultaneous JIT compilation collisions
    if rank > 0:
        waited = 0
        while not os.path.exists(os.path.join(marker_dir, f"worker{rank-1}")) and waited < 900:
            time.sleep(5)
            waited += 5

    from arc_solver import worker

    with open(os.path.join(marker_dir, f"worker{rank}"), "w") as f:
        f.write("Ok")

    print(f"[Rank {rank}] start!")

    attempts = 0
    while attempts < 2 and time.time() < end_time:
        attempts += 1
        try:
            worker(rank, queue, end_time, test_path=test_path)
            break
        except Exception as e:
            print(f"[Rank {rank}] worker crashed ({type(e).__name__}: {e}); attempt {attempts}")
            traceback.print_exc()
            try:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
            if attempts >= 2:
                print(f"[Rank {rank}] giving up.")

    print(f"[Rank {rank}] done!")


def estimated_work(task):
    """Token cost proxy for ordering tasks cheap-first."""
    def ntok(g):
        return len(g) * (len(g[0]) + 1)
    train_tokens = sum(ntok(p["input"]) + ntok(p["output"]) for p in task["train"])
    ratios = [ntok(p["output"]) / max(1, ntok(p["input"])) for p in task["train"]]
    ratios.sort()
    ratio = ratios[len(ratios) // 2]
    test_tokens = sum(ntok(t["input"]) * (1 + ratio) for t in task["test"])
    return train_tokens * 16 + test_tokens * 8 * len(task["test"])


def run_symbolic_prepass(data, keys, symbolic_out_dir=None):
    """
    Executes fast symbolic pre-pass on CPU.
    Returns:
      solved_tasks: list of task keys solved with 100% exact demonstration verification.
    """
    if symbolic_out_dir is None:
        symbolic_out_dir = os.getenv("ARC_SYMBOLIC_DIR", "/kaggle/working/symbolic_outputs")
        # fallback to local if /kaggle not writable
        try:
            os.makedirs(symbolic_out_dir, exist_ok=True)
        except PermissionError:
            symbolic_out_dir = "./symbolic_outputs"
            os.makedirs(symbolic_out_dir, exist_ok=True)
    else:
        try:
            os.makedirs(symbolic_out_dir, exist_ok=True)
        except PermissionError:
            symbolic_out_dir = "./symbolic_outputs"
            os.makedirs(symbolic_out_dir, exist_ok=True)
    solved_tasks = []
    symbolic_summary = {}

    t0 = time.time()
    for k in keys:
        task = data[k]
        res = solve_task_symbolic(task)
        if res["has_exact"]:
            solved_tasks.append(k)
            preds = res["exact_predictions"]
            symbolic_summary[k] = {
                "rule": res["exact_rule_name"],
                "predictions": [p.tolist() if isinstance(p, np.ndarray) else p for p in preds]
            }

            # Save per-test-item candidate
            for i, p in enumerate(preds):
                subkey = f"{k}_{i}"
                sample = {
                    "beam_score": 0.0,
                    "score_aug": [0.0] * 8,
                    "solution": np.asarray(p, dtype=int),
                    "is_symbolic_exact": True,
                }
                out_path = os.path.join(symbolic_out_dir, f"{subkey}.symbolic")
                with bz2.BZ2File(out_path, "w") as f:
                    pickle.dump([sample], f)

    elapsed = time.time() - t0
    print(f"[Symbolic Pre-Pass] checked {len(keys)} tasks in {elapsed:.2f}s: {len(solved_tasks)} solved with exact verified rules!")
    summary_path = os.path.join(os.path.dirname(symbolic_out_dir), "symbolic_summary.json")
    with open(summary_path, "w") as f:
        json.dump(symbolic_summary, f, indent=2)

    return solved_tasks


def resolve_challenges_path(rerun=False):
    fname = "arc-agi_test_challenges.json" if rerun else "arc-agi_evaluation_challenges.json"
    alt_fname = "arc_agi_test_challenges.json" if rerun else "arc_agi_evaluation_challenges.json"
    candidates = [
        f"/kaggle/input/competitions/arc-prize-2026-arc-agi-2/{fname}",
        f"/kaggle/input/arc-prize-2026-arc-agi-2/{fname}",
        f"/kaggle/input/arc-prize-2026/{fname}",
        f"/kaggle/input/{fname}",
        f"/kaggle/input/competitions/arc-prize-2026-arc-agi-2/{alt_fname}",
        f"/kaggle/input/arc-prize-2026-arc-agi-2/{alt_fname}",
        f"/kaggle/input/arc-prize-2026/{alt_fname}",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    import glob
    for pattern in [f"/kaggle/input/**/{fname}", f"/kaggle/input/**/{alt_fname}"]:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    return f"/kaggle/input/competitions/arc-prize-2026-arc-agi-2/{fname}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    parser.add_argument("--keys-file", type=str, default="")
    parser.add_argument("--nprocs", type=int, default=0)
    parser.add_argument("--order", type=str, default="cheap", choices=["cheap", "sorted", "file"])
    parser.add_argument("--test-path", type=str, default="")
    parser.add_argument("--marker-dir", type=str, default="/kaggle/working/markers")
    parser.add_argument("--skip-symbolic", action="store_true", default=False)
    args, _ = parser.parse_known_args()

    rerun_mode = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))

    if args.test_path:
        test_path = args.test_path
    elif rerun_mode:
        test_path = resolve_challenges_path(rerun=True)
    else:
        test_path = resolve_challenges_path(rerun=False)

    with open(test_path, "r") as f:
        data = json.load(f)

    if args.keys_file:
        with open(args.keys_file) as f:
            keys = [k for k in json.load(f) if k in data]
    else:
        keys = sorted(data.keys())
        # NOTE: debug key filter removed — ARC_DEBUG_KEYS only applied when explicitly set
        # KAGGLE_IS_COMPETITION_RERUN is NOT set during user "Save & Run" submissions,
        # only during the official competition rerun (auto-grading). Previously this
        # block was filtering ALL manual submissions to only 4 debug tasks, causing
        # 32-min runs instead of 12-hour full 240-task runs.
        debug_keys_env = os.getenv("ARC_DEBUG_KEYS", "")
        if debug_keys_env:
            debug_keys = [k.strip() for k in debug_keys_env.split(",") if k.strip()]
            keys = [k for k in keys if k in debug_keys]

    # 1. Run Symbolic Pre-Pass
    if not args.skip_symbolic:
        solved_tasks = run_symbolic_prepass(data, keys)
        if solved_tasks:
            orig_len = len(keys)
            keys = [k for k in keys if k not in set(solved_tasks)]
            print(f"[starter] Excluded {len(solved_tasks)} symbolically solved tasks from GPU queue ({orig_len} -> {len(keys)} remaining)")

    # 2. Sort remaining tasks
    if args.order == "cheap":
        keys = sorted(keys, key=lambda k: estimated_work(data[k]))
    elif args.order == "sorted":
        keys = sorted(keys)

    # 2b. Resume: skip tasks already having all shards in out_dir/symbolic (power outage safe)
    try:
        out_dir_resume = os.getenv("ARC_OUT_DIR", "/kaggle/inference_outputs")
        sym_dir_resume = "/kaggle/working/symbolic_outputs"
        if not os.path.isdir(out_dir_resume) and os.path.isdir("./inference_outputs"):
            out_dir_resume = "./inference_outputs"
        have_files = set()
        if os.path.isdir(out_dir_resume):
            have_files.update(os.listdir(out_dir_resume))
        if os.path.isdir(sym_dir_resume):
            have_files.update(os.listdir(sym_dir_resume))
        if os.path.isdir("./symbolic_outputs"):
            have_files.update(os.listdir("./symbolic_outputs"))
        if have_files:
            done = []
            remaining_keys = []
            for k in keys:
                n_out = len(data[k]["test"])
                is_done = all(any(f.startswith(f"{k}_{i}") for f in have_files) for i in range(n_out))
                if is_done:
                    done.append(k)
                else:
                    remaining_keys.append(k)
            if done:
                print(f"[resume] skip {len(done)} already done ({len(done)}/{len(done)+len(remaining_keys)}), {len(remaining_keys)} remaining")
                keys = remaining_keys
    except Exception as e:
        print(f"[resume] check failed: {e}")

    nprocs = args.nprocs or min(4, max(1, torch.cuda.device_count()))
    # marker_dir fallback to local if /kaggle not writable
    try:
        os.makedirs(args.marker_dir, exist_ok=True)
    except PermissionError:
        args.marker_dir = "./markers"
        os.makedirs(args.marker_dir, exist_ok=True)
    for f_ in os.listdir(args.marker_dir):
        try:
            os.remove(os.path.join(args.marker_dir, f_))
        except Exception:
            pass

    print(f"[starter] {len(keys)} tasks, {nprocs} workers, order={args.order}, "
          f"budget={(args.end_time - time.time())/60:.1f} min, test_path={test_path}")

    queue = mp.Manager().Queue()
    for key in keys:
        queue.put(key)
    for _ in range(nprocs):
        queue.put(None)

    try:
        mp.spawn(local_worker, args=(queue, args.end_time, test_path, args.marker_dir), nprocs=nprocs)
    except Exception as e:
        print(f"[starter] spawn finished with error: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("[starter] finished with FAILURE.")
        sys.exit(1)  # FAIL-FAST: nonzero rc so notebook cells abort instead of continuing
    print("[starter] finished.")
