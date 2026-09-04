"""
arc_invariants.py - ARC-specific structural & invariant verification engine
"""
import numpy as np
from collections import Counter
from typing import List, Tuple, Optional, Set, Dict, Any

def shape(grid) -> Tuple[int, int]:
    if isinstance(grid, np.ndarray):
        return grid.shape[0], grid.shape[1]
    if not isinstance(grid, (list, tuple)) or len(grid) == 0:
        return 0, 0
    return len(grid), len(grid[0]) if isinstance(grid[0], (list, tuple, np.ndarray)) else 0

def is_valid_grid(grid) -> bool:
    """Strictly validates if a grid is a valid ARC 2D discrete grid (max 30x30, ints 0-9)."""
    if grid is None:
        return False
    if isinstance(grid, np.ndarray):
        if grid.ndim != 2:
            return False
        h, w = grid.shape
        if h == 0 or w == 0 or h > 30 or w > 30:
            return False
        if not np.issubdtype(grid.dtype, np.integer):
            return False
        if grid.min() < 0 or grid.max() > 9:
            return False
        return True
    elif isinstance(grid, list):
        if len(grid) == 0 or len(grid) > 30:
            return False
        if not all(isinstance(row, list) and len(row) > 0 for row in grid):
            return False
        w = len(grid[0])
        if w == 0 or w > 30:
            return False
        for row in grid:
            if len(row) != w:
                return False
            for val in row:
                if not isinstance(val, (int, np.integer)) or not (0 <= val <= 9):
                    return False
        return True
    return False

def to_np_grid(grid) -> Optional[np.ndarray]:
    if grid is None:
        return None
    if isinstance(grid, np.ndarray):
        if is_valid_grid(grid):
            return grid.astype(int)
        return None
    if is_valid_grid(grid):
        return np.array(grid, dtype=int)
    return None

def infer_shape_rule(train_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Infers the mathematical relationship between input grid shapes and output grid shapes.
    Possible rules:
    - 'same': Output shape is identical to input shape (H_out == H_in, W_out == W_in)
    - 'constant': Output shape is constant across all examples (e.g. 3x3)
    - 'scale': Output shape is a fixed integer multiple of input shape (e.g. 2x, 3x)
    - 'fraction': Output shape is a fixed fraction (e.g. 1/2, 1/3)
    - 'transpose': Output shape is transposed (H_out == W_in, W_out == H_in)
    - 'unknown': Variable/dynamic shape
    """
    if not train_pairs:
        return {"rule": "unknown"}

    in_shapes = [shape(p["input"]) for p in train_pairs]
    out_shapes = [shape(p["output"]) for p in train_pairs]

    # Check 'same'
    if all(si == so for si, so in zip(in_shapes, out_shapes)):
        return {"rule": "same"}

    # Check 'constant'
    if len(set(out_shapes)) == 1:
        return {"rule": "constant", "shape": out_shapes[0]}

    # Check 'scale'
    h_ratios = [so[0] / si[0] if si[0] > 0 else 0 for si, so in zip(in_shapes, out_shapes)]
    w_ratios = [so[1] / si[1] if si[1] > 0 else 0 for si, so in zip(in_shapes, out_shapes)]
    if len(set(h_ratios)) == 1 and len(set(w_ratios)) == 1:
        hr, wr = h_ratios[0], w_ratios[0]
        if hr > 0 and wr > 0:
            return {"rule": "scale", "scale_h": hr, "scale_w": wr}

    # Check 'transpose'
    if all(si[0] == so[1] and si[1] == so[0] for si, so in zip(in_shapes, out_shapes)):
        return {"rule": "transpose"}

    return {"rule": "unknown"}

def predict_output_shape(shape_rule: Dict[str, Any], test_input) -> Optional[Tuple[int, int]]:
    """Predicts expected output shape given the inferred shape rule and test input."""
    hin, win = shape(test_input)
    rule = shape_rule.get("rule", "unknown")
    if rule == "same":
        return hin, win
    elif rule == "constant":
        return shape_rule["shape"]
    elif rule == "scale":
        sh = int(round(hin * shape_rule["scale_h"]))
        sw = int(round(win * shape_rule["scale_w"]))
        if 0 < sh <= 30 and 0 < sw <= 30:
            return sh, sw
    elif rule == "transpose":
        return win, hin
    return None

def infer_allowed_palette(train_pairs: List[Dict[str, Any]], test_input) -> Set[int]:
    """
    Determines the set of permitted colors for the test output.
    In almost all ARC tasks, the output colors must be:
    - Colors appearing in the test input, OR
    - Fixed constant colors that appeared across all training outputs.
    """
    test_in_colors = set(np.unique(np.asarray(test_input))) if isinstance(test_input, (list, np.ndarray)) else set()
    train_out_colors = set()
    for p in train_pairs:
        train_out_colors.update(np.unique(np.asarray(p["output"])))

    # The allowed palette is the union of test input colors and training output colors
    return test_in_colors | train_out_colors

def validate_candidate_invariants(
    candidate: np.ndarray,
    train_pairs: List[Dict[str, Any]],
    test_input: np.ndarray,
    shape_rule: Optional[Dict[str, Any]] = None
) -> Tuple[bool, float]:
    """
    Evaluates how well a candidate adheres to ARC structural invariants.
    Returns:
      (is_plausible: bool, penalty_score: float)
    """
    if not is_valid_grid(candidate):
        return False, 1000.0

    penalty = 0.0
    cand_h, cand_w = candidate.shape

    if shape_rule is None:
        shape_rule = infer_shape_rule(train_pairs)

    expected_shape = predict_output_shape(shape_rule, test_input)
    if expected_shape is not None:
        exp_h, exp_w = expected_shape
        if (cand_h, cand_w) != (exp_h, exp_w):
            # Heavy penalty for shape mismatch when shape rule is clear
            penalty += 50.0 + 5.0 * (abs(cand_h - exp_h) + abs(cand_w - exp_w))

    # Palette validation
    allowed_palette = infer_allowed_palette(train_pairs, test_input)
    cand_colors = set(np.unique(candidate))
    disallowed_colors = cand_colors - allowed_palette
    if disallowed_colors:
        # Severe penalty for hallucinating colors not present in problem
        penalty += 30.0 * len(disallowed_colors)

    return penalty < 40.0, penalty

def grid_hamming_distance(grid1: np.ndarray, grid2: np.ndarray) -> float:
    """Computes normalized distance between two grids for attempt diversification."""
    if grid1.shape != grid2.shape:
        return 1.0
    if grid1.size == 0:
        return 0.0
    return float(np.mean(grid1 != grid2))
