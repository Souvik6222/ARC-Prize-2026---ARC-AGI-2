# Step-by-Step Execution Guide: What `arc2-nvarc-v1.ipynb` Does

This document walks through the exact step-by-step sequence of events that occurs when you run the notebook **`arc2-nvarc-v1.ipynb`** from top to bottom.

---

## 🧭 High-Level Execution Timeline

```
 [Step 1: Timer & Budget] ────────► [Step 2: Hardware & Environment Setup]
            │
            ▼
 [Step 3: Write arc_loader.py] ───► [Step 4: Write arc_decoder.py]
            │
            ▼
 [Step 5: Write arc_solver.py] ───► [Step 6: Write starter.py]
            │
            ▼
 [Step 7: Phase 1 — Main Execution Pass (Multi-GPU LoRA + DFS)]
            │
            ▼
 [Step 8: Phase 2 — Adaptive Catch-Up & Deep Search (Spare Time)]
            │
            ▼
 [Step 9: Phase 3 — Candidate Re-ranking & submission.json Creation]
```

---

## 📌 Detailed Step-by-Step Walkthrough

---

### Step 1: Initialize Global Timer & Budget (Cell 1)
**What happens:**
1. The script reads the environment variable `KAGGLE_IS_COMPETITION_RERUN` to determine if it is running in Kaggle's hidden test evaluation.
2. It sets a strict countdown clock:
   - **Total Allowed Time**: 12 hours ($43,200$ seconds).
   - **Write Buffer**: 10 minutes ($600$ seconds reserved for final JSON assembly).
   - **`global_end_time`**: Calculated as `start_time + 12h - 10m`.
3. If running interactively (e.g., debug mode), it sets a short 5-minute budget so that developers can test the entire pipeline quickly without waiting 12 hours.

---

### Step 2: Environment Cleaning & Hardware Verification (Cell 2)
**What happens:**
1. Runs `!pip uninstall -y tensorflow` to remove TensorFlow and avoid CUDA memory fragmentation or version conflicts.
2. Queries PyTorch to check available GPUs (typically 4 $\times$ NVIDIA L4 GPUs).
3. Resolves file paths for the competition inputs and pretrained Qwen3-4B model weights.
4. Creates a working log folder at `/kaggle/working/logs`.

---

### Step 3: Write Helper Module 1 — `arc_loader.py` (Cell 3)
**What happens:**
Creates `arc_loader.py` in the workspace. This module contains all data transformation and prompt engineering functions:
1. **`convert_grid_to_string()`**: Serializes 2D numerical arrays (e.g. `[[0, 1], [2, 3]]`) into newline-separated text strings (`"01\n23"`).
2. **`permute_mod()`**: Generates geometric transformations (8 symmetries: 4 rotations $\times$ 2 reflections) and color permutations (shuffling numbers 0–9).
3. **`QwenFormatter`**: Formats puzzle demonstration pairs into the Qwen ChatML structure:
   ```text
   <|im_start|>user
   01
   23<|im_end|><|im_start|>assistant
   10
   32<|im_end|>
   ```
4. **`convert_tokens_to_array()`**: Parses the model's generated token stream back into clean 2D NumPy matrices.

---

### Step 4: Write Helper Module 2 — `arc_decoder.py` (Cell 4)
**What happens:**
Creates `arc_decoder.py` in the workspace. This module is responsible for candidate aggregation and voting:
1. **`score_kgmon(guesses)`**:
   Computes the consensus score for every candidate grid:
   $$\text{Score} = \text{Vote Count (Frequency)} - \text{Mean NLL on Augmented Views}$$
2. **`ArcDecoder` Class**:
   - Scans output folders for generated candidate grids.
   - Groups identical grids together.
   - Sorts candidates by `score_kgmon`.
   - Returns the top 2 best solutions per puzzle.

---

### Step 5: Write Helper Module 3 — `arc_solver.py` (Cell 5)
**What happens:**
Creates `arc_solver.py`, the core neural reasoning and search engine:
1. **Model Loading**: Uses `Unsloth` (`FastLanguageModel`) to load **Qwen3-4B** in `bfloat16` precision and attaches a LoRA adapter ($r=256$, $\alpha=32$).
2. **`turbo_dfs()`**: Implements a constrained Depth-First Search tree exploration algorithm with KV-cache:
   - Restricts token generation strictly to valid ARC tokens (`0`–`9`, `\n`, `<|im_end|>`).
   - Prunes any branch whose log-probability drops below a minimum threshold (`min_prob`).
3. **`worker()` Loop**: The main processing routine executed on each individual GPU for each task.

---

### Step 6: Write Helper Module 4 — `starter.py` (Cell 6)
**What happens:**
Creates `starter.py`, which manages multi-GPU multiprocessing:
1. **`estimated_work(task)`**:
   Calculates a token-size complexity score for every puzzle.
2. **Cheap-First Sorting**:
   Sorts all test puzzles so the smallest/simplest puzzles are solved first. This ensures maximum puzzle throughput before the 12-hour deadline.
3. **`mp.spawn`**:
   Spawns 4 worker processes (one dedicated process per GPU: GPU 0, 1, 2, 3) communicating via a shared task queue.

---

### Step 7: Execute Phase 1 — The Main Execution Pass (Cell 7)
**What happens:**
The notebook launches `starter.py` as a subprocess to solve all tasks across all GPUs.

```
                  ┌─────────────────────────────────────────┐
                  │          Task Queue (Shared)            │
                  │  [Task A] [Task B] [Task C] [Task D]... │
                  └────┬──────────┬──────────┬──────────┬───┘
                       │          │          │          │
                       ▼          ▼          ▼          ▼
                   [GPU 0]    [GPU 1]    [GPU 2]    [GPU 3]
                   Worker 0   Worker 1   Worker 2   Worker 3
```

#### Inside Each GPU Worker for Every Single Puzzle:
1. **LoRA Reset**: Clears any previous task adaptations and resets the LoRA weights to their clean baseline state.
2. **Demonstration Augmentation**: Takes the 2–5 demonstration pairs and generates **16 augmented variations** (rotations, flips, color swaps).
3. **Test-Time Training (TTT)**: Fine-tunes the model on these 16 augmented demonstrations for **1 epoch** using Unsloth (takes ~2–5 seconds).
4. **Evaluation Augmentation**: Takes the test input grid and creates augmented views (e.g. rotated $90^\circ, 180^\circ$, flipped, etc.).
5. **Turbo DFS Decoding**: Runs constrained tree search to generate candidate output tokens for each view.
6. **Inversion**: Inverts each generated grid back to its original orientation and color mapping.
7. **Validation Scoring**: Computes the model's cross-entropy loss (NLL) on augmented views to measure how confident the model is in that solution.
8. **Checkpointing**: Saves the candidate grids and their scores into `/kaggle/inference_outputs/<task_id>.pkl.bz2`.

---

### Step 8: Execute Phase 2 — Adaptive Catch-Up & Deep Search (Cell 8)
**What happens:**
Once Phase 1 finishes, the notebook checks how much time remains before the 12-hour limit:

1. **Time Check**:
   - If **less than 20 minutes remain**, Phase 2 is skipped to protect the submission buffer.
   - If **more than 20 minutes remain**, it analyzes all completed outputs.
2. **Catch-Up Pass**:
   If any puzzles were skipped in Phase 1 due to temporary bottlenecks, it processes them first.
3. **Deep Search Pass on Starved Tasks**:
   - Finds tasks that produced fewer than 2 distinct candidate solutions ("starved tasks").
   - Allocates the remaining budget evenly across these difficult puzzles.
   - Runs a wider, deeper search:
     - **24 augmented views** (instead of 16).
     - **Lower probability threshold** (`min_prob=0.1` for wider exploration).
     - **Extended DFS timeout** (up to 2400 seconds per puzzle).
   - Saves deep results into `/kaggle/inference_outputs_deep/`.

---

### Step 9: Execute Phase 3 — Candidate Re-ranking & Submission Assembly (Cell 9)
**What happens:**
1. **Result Pooling**:
   Loads all candidate files from both Phase 1 (`/kaggle/inference_outputs`) and Phase 2 (`/kaggle/inference_outputs_deep`).
2. **Consensus Ranking**:
   Runs `score_kgmon` across all pooled candidates for each puzzle to determine the top two predictions:
   - **`attempt_1`**: The highest-scoring candidate grid.
   - **`attempt_2`**: The second highest-scoring candidate grid.
3. **Schema Hardening & Fallbacks**:
   - Verifies that every prediction is a valid 2D integer matrix of size $\le 30 \times 30$.
   - If no valid prediction was generated for a puzzle, it safely falls back to the original input grid (identity transformation) or `[[0]]`.
   - Ensures `attempt_2` is not identical to `attempt_1`.
4. **Write Output File**:
   Writes the final predictions to **`/kaggle/working/submission.json`**.
5. **Integrity Check**:
   Verifies that all task IDs from the competition test set exist in the JSON file and match the required format.

---

## 📊 Summary of Data Transformations

| Stage | Input | Transformation | Output |
| :--- | :--- | :--- | :--- |
| **Data Prep** | 2D List `[[1, 2], [3, 4]]` | Serialized to string & augmented with D4 symmetries | ChatML formatted string prompt |
| **Test-Time Training** | Demonstration string prompts | Unsloth LoRA gradient descent (1 epoch) | Temporarily adapted Qwen3-4B weights |
| **Decoding** | Augmented test input prompt | Turbo DFS constrained tree search | Candidate token sequences |
| **Parsing & Inversion** | Candidate tokens | Decoded to array & geometrically inverted | Canonical 2D candidate matrices |
| **Scoring & Selection** | All candidate matrices | Multi-view consensus (`score_kgmon`) | Top-2 selected grids (`attempt_1`, `attempt_2`) |
| **Submission** | Top-2 grids for all tasks | JSON schema formatting & fallback validation | `/kaggle/working/submission.json` |
