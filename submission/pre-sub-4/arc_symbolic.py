"""
arc_symbolic.py - Fast Pure-Python/NumPy Deterministic & Symbolic ARC Rule Engine
"""
import numpy as np
from collections import Counter, deque
from typing import List, Dict, Any, Tuple, Optional, Callable

from arc_invariants import is_valid_grid, to_np_grid

def get_shape(grid: np.ndarray) -> Tuple[int, int]:
    return grid.shape[0], grid.shape[1]

def get_most_common_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))

def get_least_common_color(grid: np.ndarray, exclude_bg: bool = True) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    if exclude_bg:
        bg = get_most_common_color(grid)
        counts[bg] = 0
    non_zero = np.where(counts > 0)[0]
    if len(non_zero) == 0:
        return 0
    return int(non_zero[np.argmin(counts[non_zero])])

def get_non_bg_cells(grid: np.ndarray, bg: Optional[int] = None) -> List[Tuple[int, int]]:
    if bg is None:
        bg = get_most_common_color(grid)
    coords = np.argwhere(grid != bg)
    return [tuple(c) for c in coords]

def bbox_cells(cells: List[Tuple[int, int]]) -> Optional[Tuple[int, int, int, int]]:
    if not cells:
        return None
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    return min(rs), max(rs), min(cs), max(cs)

def crop_bbox(grid: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    r1, r2, c1, c2 = bbox
    return grid[r1:r2+1, c1:c2+1].copy()

def crop_non_bg(grid: np.ndarray, bg: Optional[int] = None) -> np.ndarray:
    if bg is None:
        bg = get_most_common_color(grid)
    cells = get_non_bg_cells(grid, bg)
    if not cells:
        return np.array([[bg]], dtype=int)
    bbox = bbox_cells(cells)
    return crop_bbox(grid, bbox)

# ----------------- Connected Components -----------------
def get_connected_components(grid: np.ndarray, bg: Optional[int] = None, diag: bool = False) -> List[Dict[str, Any]]:
    if bg is None:
        bg = get_most_common_color(grid)
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diag:
        neighbors += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dx, dy in neighbors:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] == color:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append({"color": color, "cells": cells})
    return comps

def crop_largest_component(grid: np.ndarray, bg: Optional[int] = None) -> np.ndarray:
    comps = get_connected_components(grid, bg)
    if not comps:
        return crop_non_bg(grid, bg)
    largest = max(comps, key=lambda z: len(z["cells"]))
    bbox = bbox_cells(largest["cells"])
    return crop_bbox(grid, bbox)

def crop_smallest_component(grid: np.ndarray, bg: Optional[int] = None) -> np.ndarray:
    comps = get_connected_components(grid, bg)
    if not comps:
        return crop_non_bg(grid, bg)
    smallest = min(comps, key=lambda z: len(z["cells"]))
    bbox = bbox_cells(smallest["cells"])
    return crop_bbox(grid, bbox)

# ----------------- Symmetries & Geometric Primitives -----------------
def sym_identity(g: np.ndarray) -> np.ndarray:
    return g.copy()

def sym_rot90(g: np.ndarray) -> np.ndarray:
    return np.rot90(g, k=-1).copy()

def sym_rot180(g: np.ndarray) -> np.ndarray:
    return np.rot90(g, k=2).copy()

def sym_rot270(g: np.ndarray) -> np.ndarray:
    return np.rot90(g, k=1).copy()

def sym_flip_h(g: np.ndarray) -> np.ndarray:
    return np.fliplr(g).copy()

def sym_flip_v(g: np.ndarray) -> np.ndarray:
    return np.flipud(g).copy()

def sym_transpose(g: np.ndarray) -> np.ndarray:
    return np.transpose(g).copy()

def sym_antitranspose(g: np.ndarray) -> np.ndarray:
    return np.rot90(np.flipud(g), k=1).copy()

GEOMETRIC_TRANSFORMS = [
    ("identity", sym_identity),
    ("rot90", sym_rot90),
    ("rot180", sym_rot180),
    ("rot270", sym_rot270),
    ("flip_h", sym_flip_h),
    ("flip_v", sym_flip_v),
    ("transpose", sym_transpose),
    ("antitranspose", sym_antitranspose),
]

# ----------------- Border and Subgrid Extraction -----------------
def crop_remove_border(grid: np.ndarray) -> Optional[np.ndarray]:
    h, w = grid.shape
    if h > 2 and w > 2:
        return grid[1:h-1, 1:w-1].copy()
    return None

def extract_outer_border(grid: np.ndarray) -> np.ndarray:
    bg = get_most_common_color(grid)
    h, w = grid.shape
    out = np.full_like(grid, bg)
    out[0, :] = grid[0, :]
    out[h-1, :] = grid[h-1, :]
    out[:, 0] = grid[:, 0]
    out[:, w-1] = grid[:, w-1]
    return out

# ----------------- Symmetry Completion -----------------
def complete_symmetry_h(grid: np.ndarray) -> np.ndarray:
    """Completes horizontal symmetry by overlaying left half or right half."""
    h, w = grid.shape
    mid = w // 2
    out = grid.copy()
    # Mirror left to right
    for r in range(h):
        for c in range(mid):
            if out[r, c] != 0 and out[r, w - 1 - c] == 0:
                out[r, w - 1 - c] = out[r, c]
            elif out[r, w - 1 - c] != 0 and out[r, c] == 0:
                out[r, c] = out[r, w - 1 - c]
    return out

def complete_symmetry_v(grid: np.ndarray) -> np.ndarray:
    """Completes vertical symmetry by overlaying top half or bottom half."""
    h, w = grid.shape
    mid = h // 2
    out = grid.copy()
    for r in range(mid):
        for c in range(w):
            if out[r, c] != 0 and out[h - 1 - r, c] == 0:
                out[h - 1 - r, c] = out[r, c]
            elif out[h - 1 - r, c] != 0 and out[r, c] == 0:
                out[r, c] = out[h - 1 - r, c]
    return out

# ----------------- Color Remapping -----------------
def replace_colors(grid: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    out = grid.copy()
    for src, dst in mapping.items():
        out[grid == src] = dst
    return out

def infer_exact_color_map(train_pairs: List[Dict[str, Any]]) -> Optional[Dict[int, int]]:
    mapping = {}
    for p in train_pairs:
        inp = np.asarray(p["input"])
        out = np.asarray(p["output"])
        if inp.shape != out.shape:
            return None
        diff_mask = inp != out
        if not np.any(diff_mask):
            continue
        unique_pairs = np.unique(np.stack([inp.ravel(), out.ravel()], axis=1), axis=0)
        for src, dst in unique_pairs:
            src, dst = int(src), int(dst)
            if src in mapping and mapping[src] != dst:
                return None
            mapping[src] = dst
    return mapping if mapping else None

# ----------------- Tiling & Scaling -----------------
def tile_grid(grid: np.ndarray, tr: int, tc: int) -> np.ndarray:
    return np.tile(grid, (tr, tc))

def scale_grid(grid: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(grid, np.ones((factor, factor), dtype=int))

def infer_tiling_factors(train_pairs: List[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    ratios = []
    for p in train_pairs:
        hi, wi = get_shape(np.asarray(p["input"]))
        ho, wo = get_shape(np.asarray(p["output"]))
        if hi == 0 or wi == 0 or ho % hi != 0 or wo % wi != 0:
            return None
        ratios.append((ho // hi, wo // wi))
    if ratios and len(set(ratios)) == 1:
        return ratios[0]
    return None

def infer_scale_factor(train_pairs: List[Dict[str, Any]]) -> Optional[int]:
    scales = []
    for p in train_pairs:
        hi, wi = get_shape(np.asarray(p["input"]))
        ho, wo = get_shape(np.asarray(p["output"]))
        if hi == 0 or wi == 0 or ho % hi != 0 or wo % wi != 0:
            return None
        if (ho // hi) != (wo // wi):
            return None
        scales.append(ho // hi)
    if scales and len(set(scales)) == 1 and scales[0] > 1:
        return scales[0]
    return None

# ----------------- Gravity / Drop -----------------
def apply_gravity_down(grid: np.ndarray, bg: Optional[int] = None) -> np.ndarray:
    if bg is None:
        bg = get_most_common_color(grid)
    out = np.full_like(grid, bg)
    h, w = grid.shape
    for c in range(w):
        col_vals = [int(v) for v in grid[:, c] if v != bg]
        if col_vals:
            out[h - len(col_vals):h, c] = col_vals
    return out

def apply_gravity_up(grid: np.ndarray, bg: Optional[int] = None) -> np.ndarray:
    if bg is None:
        bg = get_most_common_color(grid)
    out = np.full_like(grid, bg)
    h, w = grid.shape
    for c in range(w):
        col_vals = [int(v) for v in grid[:, c] if v != bg]
        if col_vals:
            out[:len(col_vals), c] = col_vals
    return out

def apply_gravity_right(grid: np.ndarray, bg: Optional[int] = None) -> np.ndarray:
    if bg is None:
        bg = get_most_common_color(grid)
    out = np.full_like(grid, bg)
    h, w = grid.shape
    for r in range(h):
        row_vals = [int(v) for v in grid[r, :] if v != bg]
        if row_vals:
            out[r, w - len(row_vals):w] = row_vals
    return out

def apply_gravity_left(grid: np.ndarray, bg: Optional[int] = None) -> np.ndarray:
    if bg is None:
        bg = get_most_common_color(grid)
    out = np.full_like(grid, bg)
    h, w = grid.shape
    for r in range(h):
        row_vals = [int(v) for v in grid[r, :] if v != bg]
        if row_vals:
            out[r, :len(row_vals)] = row_vals
    return out

# ----------------- Summary / 1x1 Extraction -----------------
def extract_most_common_non_bg(grid: np.ndarray) -> np.ndarray:
    bg = get_most_common_color(grid)
    non_bg = grid[grid != bg]
    if len(non_bg) == 0:
        return np.array([[bg]], dtype=int)
    c = Counter(non_bg.tolist()).most_common(1)[0][0]
    return np.array([[c]], dtype=int)

def extract_least_common_non_bg(grid: np.ndarray) -> np.ndarray:
    bg = get_most_common_color(grid)
    non_bg = grid[grid != bg]
    if len(non_bg) == 0:
        return np.array([[bg]], dtype=int)
    c = Counter(non_bg.tolist()).most_common()[-1][0]
    return np.array([[c]], dtype=int)

def extract_component_count(grid: np.ndarray) -> np.ndarray:
    comps = get_connected_components(grid)
    return np.array([[min(9, len(comps))]], dtype=int)

def extract_unique_color_count(grid: np.ndarray) -> np.ndarray:
    bg = get_most_common_color(grid)
    unique_non_bg = len(set(grid.ravel()) - {bg})
    return np.array([[min(9, unique_non_bg)]], dtype=int)


# ----------------- Rule Candidate Class -----------------
class SymbolicRule:
    def __init__(self, name: str, fn: Callable[[np.ndarray], Optional[np.ndarray]], priority: int = 50):
        self.name = name
        self.fn = fn
        self.priority = priority

    def apply(self, grid: np.ndarray) -> Optional[np.ndarray]:
        try:
            res = self.fn(grid)
            if is_valid_grid(res):
                return res
            return None
        except Exception:
            return None

    def verifies_all(self, train_pairs: List[Dict[str, Any]]) -> bool:
        for p in train_pairs:
            inp = np.asarray(p["input"])
            expected = np.asarray(p["output"])
            pred = self.apply(inp)
            if pred is None or not np.array_equal(pred, expected):
                return False
        return True


def build_symbolic_rules(train_pairs: List[Dict[str, Any]]) -> List[SymbolicRule]:
    """Generates all deterministic/symbolic candidate transformations for a task."""
    rules = []

    # 1. Geometric Symmetries
    for name, fn in GEOMETRIC_TRANSFORMS:
        prio = 100 if name == "identity" else 75
        rules.append(SymbolicRule(name, fn, priority=prio))

    # 2. Crop Non-Background & Components
    rules.append(SymbolicRule("crop_non_bg", crop_non_bg, priority=80))
    rules.append(SymbolicRule("crop_largest_comp", crop_largest_component, priority=75))
    rules.append(SymbolicRule("crop_smallest_comp", crop_smallest_component, priority=70))
    rules.append(SymbolicRule("crop_remove_border", crop_remove_border, priority=65))
    rules.append(SymbolicRule("extract_outer_border", extract_outer_border, priority=60))

    # 3. Symmetry completions
    rules.append(SymbolicRule("complete_symmetry_h", complete_symmetry_h, priority=60))
    rules.append(SymbolicRule("complete_symmetry_v", complete_symmetry_v, priority=60))

    # 4. Crop non-bg + Symmetries
    for geo_name, geo_fn in GEOMETRIC_TRANSFORMS:
        if geo_name != "identity":
            rules.append(SymbolicRule(
                f"crop_non_bg_then_{geo_name}",
                lambda g, fn=geo_fn: fn(crop_non_bg(g)),
                priority=68
            ))

    # 5. Color Mapping
    color_map = infer_exact_color_map(train_pairs)
    if color_map is not None:
        rules.append(SymbolicRule(
            f"exact_color_map_{color_map}",
            lambda g, m=color_map: replace_colors(g, m),
            priority=90
        ))
        rules.append(SymbolicRule(
            f"crop_non_bg_then_color_map_{color_map}",
            lambda g, m=color_map: replace_colors(crop_non_bg(g), m),
            priority=85
        ))
        for geo_name, geo_fn in GEOMETRIC_TRANSFORMS:
            rules.append(SymbolicRule(
                f"{geo_name}_then_color_map_{color_map}",
                lambda g, fn=geo_fn, m=color_map: replace_colors(fn(g), m),
                priority=80
            ))

    # 6. Tiling
    tiling = infer_tiling_factors(train_pairs)
    if tiling is not None:
        tr, tc = tiling
        rules.append(SymbolicRule(f"tile_{tr}x{tc}", lambda g, tr=tr, tc=tc: tile_grid(g, tr, tc), priority=75))
        for geo_name, geo_fn in GEOMETRIC_TRANSFORMS:
            rules.append(SymbolicRule(
                f"{geo_name}_then_tile_{tr}x{tc}",
                lambda g, fn=geo_fn, tr=tr, tc=tc: tile_grid(fn(g), tr, tc),
                priority=70
            ))

    # 7. Scaling
    scale = infer_scale_factor(train_pairs)
    if scale is not None:
        rules.append(SymbolicRule(f"scale_{scale}x", lambda g, s=scale: scale_grid(g, s), priority=70))

    # 8. Gravity
    rules.append(SymbolicRule("gravity_down", apply_gravity_down, priority=50))
    rules.append(SymbolicRule("gravity_up", apply_gravity_up, priority=50))
    rules.append(SymbolicRule("gravity_right", apply_gravity_right, priority=50))
    rules.append(SymbolicRule("gravity_left", apply_gravity_left, priority=50))

    # 9. 1x1 Summaries
    rules.append(SymbolicRule("most_common_non_bg", extract_most_common_non_bg, priority=35))
    rules.append(SymbolicRule("least_common_non_bg", extract_least_common_non_bg, priority=35))
    rules.append(SymbolicRule("component_count", extract_component_count, priority=35))
    rules.append(SymbolicRule("unique_color_count", extract_unique_color_count, priority=35))

    return rules


def solve_task_symbolic(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates all symbolic rule candidates against the task demonstration pairs.
    Returns:
      {
        'has_exact': bool,
        'exact_rule_name': Optional[str],
        'exact_rule': Optional[SymbolicRule],
        'exact_predictions': List[np.ndarray],
        'top_symbolic_candidates': List[Tuple[str, np.ndarray, float]] # (name, grid, score)
      }
    """
    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    rules = build_symbolic_rules(train_pairs)
    exact_rules = []

    for rule in rules:
        if rule.verifies_all(train_pairs):
            exact_rules.append(rule)

    if exact_rules:
        # Sort by priority
        best_rule = max(exact_rules, key=lambda r: r.priority)
        preds = []
        for t in test_pairs:
            inp = np.asarray(t["input"])
            out = best_rule.apply(inp)
            preds.append(out if out is not None else inp.copy())
        return {
            "has_exact": True,
            "exact_rule_name": best_rule.name,
            "exact_rule": best_rule,
            "exact_predictions": preds,
            "top_symbolic_candidates": [(best_rule.name, p, 1000.0) for p in preds],
        }

    return {
        "has_exact": False,
        "exact_rule_name": None,
        "exact_rule": None,
        "exact_predictions": [],
        "top_symbolic_candidates": [],
    }
