"""
arc_decoder.py - Advanced V2 Candidate Scoring, Demonstration Verification, and Attempt Diversification
"""
import os
import bz2
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from arc_invariants import (
    is_valid_grid,
    validate_candidate_invariants,
    infer_shape_rule,
    grid_hamming_distance,
)

def hashable(guess) -> Tuple[Tuple[int, ...], ...]:
    return tuple(map(tuple, guess))

def _valid_sample(sample: Dict[str, Any]) -> bool:
    try:
        sol = np.asarray(sample["solution"])
        if not (isinstance(sol, np.ndarray) and sol.ndim == 2 and all(0 < x <= 30 for x in sol.shape)):
            return False
        if not np.isfinite(sample.get("beam_score", 0.0)):
            return False
        score_aug = sample.get("score_aug", [])
        if len(score_aug) and not np.all(np.isfinite(score_aug)):
            return False
        return True
    except Exception:
        return False

# ----------------- Scoring Algorithms -----------------

def score_kgmon(guesses: Dict[str, Any]) -> List[np.ndarray]:
    """Baseline kgmon consensus scoring: Vote Count - Mean Augmented NLL."""
    guess_list = list(guesses.values())
    scores = {}
    for g in guess_list:
        h = hashable(g["solution"])
        x = scores.setdefault(h, [[], g["solution"]])
        x[0].append(g)

    ranked = []
    for sc_list, sol in scores.values():
        inf_score = len(sc_list)
        aug_nlls = [np.mean(g["score_aug"]) for g in sc_list if len(g.get("score_aug", []))]
        mean_aug = float(np.mean(aug_nlls)) if aug_nlls else 0.0
        score = inf_score - mean_aug
        ranked.append((score, sol))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in ranked]


def score_v2(guesses: Dict[str, Any], task_train: Optional[List[Dict[str, Any]]] = None, test_input: Optional[np.ndarray] = None) -> List[np.ndarray]:
    """
    V2 Enhanced Consensus Scoring:
    - Vote Count - Mean Augmented NLL
    - Symbolic exact rule bonus (+1000.0)
    - Structural invariant penalties (shape mismatch & color hallucination)
    """
    guess_list = list(guesses.values())
    scores = {}
    for g in guess_list:
        h = hashable(g["solution"])
        x = scores.setdefault(h, [[], g["solution"], g.get("is_symbolic_exact", False)])
        x[0].append(g)
        if g.get("is_symbolic_exact", False):
            x[2] = True

    shape_rule = infer_shape_rule(task_train) if task_train else None

    ranked = []
    for sc_list, sol, is_exact in scores.values():
        sol_arr = np.asarray(sol)
        inf_score = len(sc_list)
        aug_nlls = [np.mean(g["score_aug"]) for g in sc_list if len(g.get("score_aug", []))]
        mean_aug = float(np.mean(aug_nlls)) if aug_nlls else 0.0
        base_score = inf_score - mean_aug

        # Exact verified symbolic rule bonus
        if is_exact:
            base_score += 1000.0

        # Structural invariant check
        if task_train and test_input is not None:
            is_plausible, penalty = validate_candidate_invariants(sol_arr, task_train, test_input, shape_rule)
            base_score -= penalty

        ranked.append((base_score, sol_arr))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in ranked]


def select_orthogonal_top2(
    ordered_candidates: List[np.ndarray],
    fallback_input: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensures attempt_1 and attempt_2 represent diverse hypotheses.
    Attempt 1: Best overall candidate.
    Attempt 2: Best alternative candidate with non-zero structural distance.
    """
    if not ordered_candidates:
        fb = fallback_input if fallback_input is not None else np.array([[0]], dtype=int)
        return fb, np.array([[0]], dtype=int)

    attempt_1 = ordered_candidates[0]
    attempt_2 = None

    for cand in ordered_candidates[1:]:
        if not np.array_equal(cand, attempt_1):
            attempt_2 = cand
            break

    if attempt_2 is None:
        attempt_2 = np.array([[0]], dtype=int)

    return attempt_1, attempt_2


class ArcDecoder:

    def __init__(self, dataset, n_guesses: int = 2):
        self.dataset = dataset
        self.n_guesses = n_guesses
        self.decoded_results = {}
        try:
            self.valid_keys = set(dataset.keys)
        except Exception:
            self.valid_keys = None

    def load_decoded_results(self, store: str, run_name: str = "") -> int:
        if not os.path.isdir(store):
            print(f"*** No decoded results at {store}")
            return 0
        n_files = n_samples = n_bad = 0
        for key in os.listdir(store):
            if key.startswith(".") or key.endswith((".json", ".txt", ".tmp")):
                continue
            try:
                with bz2.BZ2File(os.path.join(store, key)) as f:
                    outputs = pickle.load(f)
            except Exception as e:
                print(f"*** Skipping corrupt shard {key}: {e}")
                n_bad += 1
                continue
            n_files += 1
            base_key = key.split(".")[0]
            if self.valid_keys is not None and base_key not in self.valid_keys and base_key.split("_")[0] not in self.valid_keys:
                n_bad += 1
                continue
            for i, sample in enumerate(outputs):
                if not _valid_sample(sample):
                    n_bad += 1
                    continue
                self.decoded_results.setdefault(base_key, {})[f"{key}{run_name}.out{i}"] = sample
                n_samples += 1
        print(f"*** Loaded {n_files} shards / {n_samples} samples from {store} (skipped {n_bad})")
        return n_samples

    def inject_symbolic_results(self, symbolic_dict: Dict[str, np.ndarray]):
        """Injects exact verified symbolic candidate grids directly into decoded pool."""
        count = 0
        for base_key, pred_grid in symbolic_dict.items():
            if not is_valid_grid(pred_grid):
                continue
            sample = {
                "beam_score": 0.0,
                "score_aug": [0.0] * 8,
                "solution": np.asarray(pred_grid, dtype=int),
                "is_symbolic_exact": True,
            }
            self.decoded_results.setdefault(base_key, {})["symbolic_exact.out0"] = sample
            count += 1
        if count > 0:
            print(f"*** Injected {count} exact symbolic predictions into decoder pool")

    def candidate_stats(self) -> Dict[str, Dict[str, int]]:
        """Per output key: number of unique candidate grids and number of samples."""
        stats = {}
        for bk, v in self.decoded_results.items():
            uniq = {hashable(g["solution"]) for g in v.values()}
            stats[bk] = dict(unique=len(uniq), samples=len(v))
        return stats

    def run_selection_algo(self, selection_algorithm=score_v2) -> Dict[str, List[np.ndarray]]:
        results = {}
        for bk, v in self.decoded_results.items():
            task_train = None
            test_input = None
            try:
                task_id = bk.split("_")[0]
                task_data = self.dataset.queries.get(bk, self.dataset.queries.get(task_id))
                if task_data:
                    task_train = task_data.get("train")
                    test_input = np.asarray(task_data["test"][0]["input"])
            except Exception:
                pass

            if selection_algorithm == score_v2:
                ordered = score_v2({k: g for k, g in v.items()}, task_train=task_train, test_input=test_input)
            else:
                ordered = selection_algorithm({k: g for k, g in v.items()})

            results[bk] = ordered
        return results

    def get_diverse_attempts(self, selection_algorithm=score_v2) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        selected = self.run_selection_algo(selection_algorithm)
        attempts = {}
        for bk, cand_list in selected.items():
            fb = None
            try:
                task_id = bk.split("_")[0]
                task_data = self.dataset.queries.get(bk, self.dataset.queries.get(task_id))
                if task_data:
                    fb = np.asarray(task_data["test"][0]["input"])
            except Exception:
                pass
            att1, att2 = select_orthogonal_top2(cand_list, fallback_input=fb)
            attempts[bk] = (att1, att2)
        return attempts
