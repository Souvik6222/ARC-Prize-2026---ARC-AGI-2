"""
starter.py - Multi-GPU Orchestrator with Integrated Fast Symbolic Pre-Pass
"""
import os
# Fix6: must be set before torch CUDA init
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import sys
import time
import json
try:
    import torch
    import torch.multiprocessing as mp
except ImportError:
    torch = None
    import multiprocessing as mp

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


def run_symbolic_prepass(data, keys, symbolic_out_dir="/kaggle/working/symbolic_outputs"):
    """
    Executes fast symbolic pre-pass on CPU.
    Returns:
      solved_tasks: list of task keys solved with 100% exact demonstration verification.
    """
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
        if not rerun_mode:
            debug_keys = os.getenv("ARC_DEBUG_KEYS", "0934a4d8,36a08778,981571dc,aa4ec2a5").split(",")
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

    nprocs = args.nprocs or min(4, max(1, torch.cuda.device_count()))
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
    print("[starter] finished.")
