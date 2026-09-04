# ARC2 NVARC+ v1: Complete Beginner-Friendly Guide & Notebook Breakdown

This document provides a comprehensive, beginner-friendly explanation of the Kaggle notebook **`arc2-nvarc-v1.ipynb`**. It covers the fundamental concepts of the ARC challenge, how Large Language Models (LLMs) are applied to visual grid puzzles, and a line-by-line architectural breakdown of the code.

---

## Table of Contents
1. [The Big Picture: What is the ARC Challenge?](#1-the-big-picture-what-is-the-arc-challenge)
2. [Core Concepts & Terminology](#2-core-concepts--terminology)
3. [System Architecture & Pipeline Flow](#3-system-architecture--pipeline-flow)
4. [Component-by-Component Code Walkthrough](#4-component-by-component-code-walkthrough)
   - [Phase 0: Budget & Environment Setup (Cells 1–2)](#phase-0-budget--environment-setup-cells-12)
   - [Component 1: `arc_loader.py` — Data & Prompt Engine (Cell 3)](#component-1-arc_loaderpy--data--prompt-engine-cell-3)
   - [Component 2: `arc_decoder.py` — Candidate Scoring & Selection (Cell 4)](#component-2-arc_decoderpy--candidate-scoring--selection-cell-4)
   - [Component 3: `arc_solver.py` — Model Training & DFS Engine (Cell 5)](#component-3-arc_solverpy--model-training--dfs-engine-cell-5)
   - [Component 4: `starter.py` — Multi-GPU Orchestrator (Cell 6)](#component-4-starterpy--multi-gpu-orchestrator-cell-6)
   - [Phase 1: Primary Execution Pass (Cell 7)](#phase-1-primary-execution-pass-cell-7)
   - [Phase 2: Adaptive Catch-Up & Deep Search (Cell 8)](#phase-2-adaptive-catch-up--deep-search-cell-8)
   - [Phase 3: Final Selection & Submission Assembly (Cell 9)](#phase-3-final-selection--submission-assembly-cell-9)
5. [Summary: Why Does This Method Win Competitions?](#5-summary-why-does-this-method-win-competitions)

---

## 1. The Big Picture: What is the ARC Challenge?

The **Abstraction and Reasoning Corpus (ARC-AGI)** is a benchmark created to test an AI's ability to acquire new skills and solve novel visual reasoning problems without prior task-specific programming.

### How an ARC Task Works
- Each puzzle consists of **2D grids of numbers** (from `0` to `9`), where each number represents a distinct color (e.g., `0` = black, `1` = blue, `2` = red, etc.).
- A task provides **2 to 5 demonstration pairs** (`Input Grid` $\rightarrow$ `Output Grid`).
- You are then given a **Test Input Grid**, and your algorithm must discover the underlying rule (e.g., symmetry, path-finding, gravity, object counting, color filling) and predict the correct **Test Output Grid**.
- Competitors are allowed **2 attempts** (`attempt_1` and `attempt_2`) per test output.

```text
Demonstration 1:
Input Grid:             Output Grid:
0 1 0                   0 1 0
0 0 0     ========>     1 1 1
0 1 0                   0 1 0

Test Input:             Test Output:
0 2 0                   ? ? ?
0 0 0     ========>     ? ? ?  (Predict the matching pattern!)
0 2 0                   ? ? ?
```

---

## 2. Core Concepts & Terminology

| Concept | Explanation | Purpose in this Notebook |
| :--- | :--- | :--- |
| **Grid Serialization** | Converting a 2D matrix of numbers `[[1, 2], [3, 4]]` into plain text `"12\n34"`. | Enables standard text-based Large Language Models (e.g., `Qwen3-4B`) to process visual grid matrices. |
| **Test-Time Training (TTT)** | Fine-tuning the neural network *on the test task itself* right before generating predictions. | Adapts the model's weights specifically to the unique pattern of that individual puzzle. |
| **LoRA (Low-Rank Adaptation)** | A parameter-efficient fine-tuning method that trains small adapter matrices ($r=256$) instead of the full model. | Very fast to train (takes only a few seconds per puzzle) and easily reset back to clean weights. |
| **Test-Time Augmentation (TTA)** | Rotating ($90^\circ, 180^\circ, 270^\circ$), reflecting (horizontal/vertical flips), and permuting colors. | Provides multiple perspectives of the same puzzle. The LLM can solve an augmented view and invert the answer back. |
| **Turbo DFS (Tree Search)** | A Depth-First Search algorithm that explores possible token branches during generation. | Constrains token generation strictly to valid grid digits and newlines, pruning low-probability branches early. |
| **Score `score_kgmon`** | Ranking metric: $\text{Frequency (Votes)} - \text{Mean NLL on Augmented Views}$. | Selects the two most reliable predictions among dozens of candidate grids. |
| **Adaptive Budgeting** | Tracking the remaining time in Kaggle's 12-hour limit and dedicating spare time to harder tasks. | Maximizes solved puzzles by dynamically allocating compute time where it is most needed. |

---

## 3. System Architecture & Pipeline Flow

```mermaid
flowchart TD
    Start([Start Competition Run]) --> Load[Load Test Dataset & Filter Tasks]
    Load --> Order[Sort Tasks: Cheap/Fast First]
    Order --> MP[Spawn 4 Worker Processes on 4 GPUs]

    subgraph GPU Worker Loop [Per-Task Processing on Each GPU]
        Reset[1. Reset LoRA Weights to Baseline] --> AugTrain[2. Generate Augmented Training Pairs]
        AugTrain --> Train[3. Fine-Tune with Unsloth LoRA: 1 Epoch]
        Train --> AugEval[4. Generate Augmented Test Prompts]
        AugEval --> DFS[5. Turbo DFS Constrained Decoding]
        DFS --> Invert[6. Invert Predictions to Canonical Orientation]
        Invert --> Score[7. Compute Augmented NLL Score]
        Score --> Save[8. Save Candidates to Disk as .pkl.bz2]
    end

    MP --> GPU Worker Loop
    GPU Worker Loop --> CheckBudget{Remaining Time > 20 min?}

    CheckBudget -- Yes --> Phase2[Phase 2: Catch-Up & Deep Search on Starved Tasks]
    CheckBudget -- No --> Selection[Phase 3: Final Selection & Ensembling]
    Phase2 --> Selection

    Selection --> Top2[Score with score_kgmon & Pick Top 2 Guesses]
    Top2 --> Fallback[Apply Fallback Schema Checks]
    Fallback --> Output([Write submission.json])
```

---

## 4. Component-by-Component Code Walkthrough

### Phase 0: Budget & Environment Setup (Cells 1–2)

- **Global Wall-Clock Management**:
  - Kaggle competition submissions have a hard **12-hour (43,200s)** execution timeout.
  - The notebook sets `global_end_time = T0 + 12 * 3600 - 600` (reserving 10 minutes for data writing and schema validation).
  - Detects if it is running in interactive mode or hidden competition re-run mode via `KAGGLE_IS_COMPETITION_RERUN`.
- **Environment Cleanup**:
  - Uninstalls TensorFlow to eliminate CUDA runtime conflicts and conserve GPU memory.
  - Verifies PyTorch GPU availability (e.g., 4 $\times$ NVIDIA L4 GPUs).

---

### Component 1: `arc_loader.py` — Data & Prompt Engine (Cell 3)

This script manages grid formatting, symmetries, and dataset transformations.

#### Key Functions & Classes:
1. **`convert_grid_to_string(grid)`**:
   Transforms 2D arrays into strings separated by newlines:
   ```python
   # [[1, 2], [3, 4]] -> "12\n34"
   ```
2. **`permute_mod(a, descriptor, invert=False)`**:
   Implements dihedral D4 group symmetries (8 spatial transformations: 4 rotations $\times$ 2 reflections) and color permutations (remapping digits 0–9).
3. **`QwenFormatter`**:
   Wraps input-output pairs into Qwen chat tokens:
   ```text
   <|im_start|>user
   12
   34<|im_end|><|im_start|>assistant
   21
   43<|im_end|>
   ```
   - `convert_tokens_to_array(tokens)`: Decodes output token sequences back into NumPy 2D integer grids, verifying dimensions ($\le 30 \times 30$).
4. **`ArcDataset`**:
   In-memory data structure holding ARC tasks, handling test-time augmentations (`.augment(n=...)`) and multi-output splitting.

---

### Component 2: `arc_decoder.py` — Candidate Scoring & Selection (Cell 4)

After generating candidate grids across multiple views, this module selects the top two attempts.

#### Key Functions & Classes:
1. **`getter_kgmon(guesses)` / `score_kgmon(guesses)`**:
   Computes the consensus score for each candidate grid $g$:
   $$\text{Score}(g) = \text{Vote Count}(g) - \frac{1}{|V|} \sum_{v \in V} \text{NLL}_v(g)$$
   - **Vote Count**: How many distinct augmentations yielded this exact same grid.
   - **Mean Augmented NLL**: The average Negative Log-Likelihood (cross-entropy loss) of this solution across augmented views.
   - **Why this works**: High consensus across geometric views paired with low model uncertainty produces the most reliable prediction.
2. **`ArcDecoder`**:
   Loads compressed candidate files from `/kaggle/inference_outputs`, groups identical predictions, sorts candidates by `score_kgmon`, and returns the top 2 outputs.

---

### Component 3: `arc_solver.py` — Model Training & DFS Engine (Cell 5)

This is the core training and inference engine powered by **Unsloth** and **Qwen3-4B**.

#### 1. LoRA Configuration:
- Rank: $r = 256$, Alpha: $\alpha = 32$.
- Target Modules: All linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `embed_tokens`, `lm_head`).
- Memory Optimization: Loaded in `bfloat16`.

#### 2. `turbo_dfs(...)` (Constrained Tree Search):
- Unlike standard greedy or beam search, `turbo_dfs` explores the token decision tree using **Depth-First Search (DFS)** with KV caching.
- **Vocabulary Masking**: Constrains generation strictly to ARC tokens (digits `0-9`, newline `\n`, end token `<|im_end|>`).
- **Pruning**: Any token branch whose cumulative negative log-likelihood exceeds `max_score = -log(min_prob)` is pruned immediately.

#### 3. `worker(...)` (Per-Task Execution Loop):
For each puzzle assigned to a GPU worker:
1. **Reset LoRA**: Restores clean baseline adapter weights (`set_peft_model_state_dict`).
2. **Augment Train Data**: Creates 16 augmented demonstration views (rotations, flips, color permutations).
3. **Train (TTT)**: Fine-tunes the model for 1 epoch using `UnslothFixedTrainer`.
4. **Augment Eval Prompts**: Generates augmented versions of the test prompt.
5. **DFS Decoding**: Runs `inference_turbo_dfs` to generate candidate output tokens.
6. **Inversion**: Maps each generated grid back to the original orientation and color palette.
7. **Score Evaluation**: Evaluates cross-entropy loss (`calc_scores`) on rotated/flipped validation views.
8. **Storage**: Saves compressed candidates (`.pkl.bz2`) to disk.

---

### Component 4: `starter.py` — Multi-GPU Orchestrator (Cell 6)

1. **`estimated_work(task)`**:
   Computes a cost heuristic based on input/output grid dimensions:
   $$\text{Work} = \text{Train Tokens} \times 16 + \text{Test Tokens} \times 8 \times N_{\text{test}}$$
   - **Cheap-First Scheduling**: Sorting tasks from lowest to highest work ensures that simple, fast puzzles are solved first. A single massive puzzle will not block the queue and exhaust the 12-hour budget early.
2. **`local_worker` & `mp.spawn`**:
   Distributes the task queue evenly across all available GPUs (e.g., 4 processes for 4 GPUs).

---

### Phase 1: Primary Execution Pass (Cell 7)
- Executes `starter.py` across all tasks in "cheap-first" order.
- Utilizes the standard NVARC hyperparameter profile (`min_prob=0.15`, 16 augmentations).
- Runs until the queue is empty or the global timeout approaches.

---

### Phase 2: Adaptive Catch-Up & Deep Search (Cell 8)
If time remains ($> 20$ minutes) after Phase 1:
1. **Catch-Up**: Processes any tasks that were skipped or not reached during Phase 1.
2. **Deep Pass**: Identifies "starved" puzzles (tasks that generated fewer than 2 distinct candidate solutions) and runs a deeper search:
   - Evaluates 24 augmentations (instead of 16).
   - Lowers DFS probability threshold (`min_prob=0.1` for wider exploration).
   - Extends per-task timeout window up to 2400 seconds.

---

### Phase 3: Final Selection & Submission Assembly (Cell 9)
1. Aggregates all candidate solutions from Phase 1 and Phase 2.
2. Runs `decoder.run_selection_algo()` using `score_kgmon`.
3. Extracts `attempt_1` and `attempt_2`.
4. **Fallback & Schema Hardening**:
   - If an attempt is missing or empty, it falls back to the input grid (identity transformation) or `[[0]]`.
   - Checks that all task IDs and test queries match the competition schema.
5. Writes the final `/kaggle/working/submission.json`.

---

## 5. Summary: Why Does This Method Win Competitions?

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Key Architectural Pillars                       │
├──────────────────────────┬─────────────────────────────────────────────┤
│ 1. Test-Time Training    │ Adapts neural weights to each specific      │
│    (TTT via LoRA)        │ puzzle rule instead of relying on prompts.  │
├──────────────────────────┼─────────────────────────────────────────────┤
│ 2. Constrained DFS       │ Tree search explores valid candidate spaces │
│    Decoding              │ and eliminates grid formatting bugs.        │
├──────────────────────────┼─────────────────────────────────────────────┤
│ 3. Geometric & Color     │ Exploits natural ARC invariances across     │
│    Augmentation (TTA)    │ rotations, reflections, and color swaps.    │
├──────────────────────────┼─────────────────────────────────────────────┤
│ 4. Consensus Re-ranking  │ Combines multi-view voting with augmented   │
│    (score_kgmon)         │ NLL loss to pick the highest-fidelity grid. │
├──────────────────────────┼─────────────────────────────────────────────┤
│ 5. Adaptive Budgeting    │ Dynamic task prioritization fits maximum    │
│    & Multi-GPU Parallel  │ exploration into the 12-hour limit.         │
└──────────────────────────┴─────────────────────────────────────────────┘
```
