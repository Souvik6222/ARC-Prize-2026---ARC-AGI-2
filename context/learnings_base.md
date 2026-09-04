# ARC-AGI-2 Learnings & Knowledge Base

> **Centralized repository of validated findings, architectural insights, empirical benchmarks, and operational wisdom for the Kaggle ARC Prize 2026.**

---

## 1. Executive Summary & Benchmark Foundations

* **Competition**: ARC Prize 2026 – ARC-AGI-2 ([Kaggle Link](https://www.kaggle.com/competitions/arc-prize-2026))
* **Core Challenge**: Variable-sized 2D discrete color grids ($0$–$9$), sizes up to $30 \times 30$. Abstract reasoning and generalization from $2$–$5$ demonstration pairs per task.
* **Evaluation Metric**: Exact pixel-perfect binary matching ($1.0$ or $0.0$). No partial credit.
* **Submission Constraint**: Up to $2$ prediction attempts (`attempt_1`, `attempt_2`) per test item. A task is solved if either attempt matches the target.

---

## 2. Empirical Benchmark History

| Experiment / Run | Approach / Framework | Compute Budget | Score (LB / Local) | Key Distinguishing Feature |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** ([baseline/](file:///home/pika/Desktop/keggel_lb--ARC/baseline)) | Rule-based Cellular Automaton & Affine Search | CPU only (< 1 min) | ~3-5% baseline | Deterministic search (crop, rotate, tile, recolor) |
| **Try 1** ([try__1/](file:///home/pika/Desktop/keggel_lb--ARC/try__1)) | NVARC+ v1: Qwen3-4B + LoRA TTT + 2-Phase Adaptive Budget | 4x L4 GPUs (~11.5h) | **32.22** | Two-phase execution: cheap-first primary + adaptive deep pass |
| **Try 2** ([try__2/](file:///home/pika/Desktop/keggel_lb--ARC/try__2)) | Program076: September Anchor v2 Hardened Harness | 4x L4 GPUs (~9.5h) | **31.81** | Strict single-phase primary only + defensive fail-closed validation |

---

## 3. Core Architectural Learnings

### 3.1 Test-Time Training (TTT) with LoRA
* **Why TTT Works**: ARC tasks represent novel symbolic programs. Static pre-trained models fail to generalize zero-shot. By fine-tuning a small parameter adapter (LoRA rank 16/32, $\alpha=32$) on the few provided demonstration pairs *at test time*, the model adapts its internal priors specifically to the task's transformation rules.
* **Data Augmentation during TTT**:
  * Demonstration examples are expanded using the 8 dihedral transformations ($D_4$ group: identity, 3 rotations, 4 reflections) combined with color permutations.
  * 16 augmentations $\times$ 8 geometries = **128 training sequences per task**.
  * 1 epoch of low-learning-rate fine-tuning ($\text{LR} \approx 5\times 10^{-5}$) with Unsloth gradient checkpointing avoids catastrophic forgetting while adapting rapidly.

### 3.2 Depth-First Search (DFS) Decoding vs. Standard Generation
* **The Failure of Greedy/Nucleus Sampling**: Standard sampling often generates invalid grid dimensions, unclosed brackets, or hallucinated shapes.
* **DFS Guided Decoding**:
  * Explores the token tree step-by-step with a cumulative probability pruning threshold (`min_prob = 0.2`).
  * Enforces grid syntax validity on-the-fly (row length parity, digit range $[0, 9]$, termination tokens).
  * Returns multiple diverse candidate solutions per test view.

### 3.3 Test-Time Augmentation (TTA) & Re-Scoring (`score_kgmon`)
* Simply counting candidate frequency (majority voting) is biased towards simple, trivial outputs (e.g., all black grids or identity).
* **KGMon Scoring Formula**:
  $$\text{Score}(G) = \text{Vote Count}(G) - \lambda \cdot \text{Mean Augmented NLL}(G)$$
* High-confidence, frequently sampled candidates that also maintain high likelihood under inverted geometric augmentations receive the top rank for `attempt_1` and `attempt_2`.

### 3.4 Why Try 1 (32.22) Outperformed Try 2 (31.81): The Power of Adaptive Budgeting
* **Try 2 Bottleneck**: It ran a strict single-pass schedule. If a complex task did not yield 2 confident candidates in Phase 1, it received generic fallbacks.
* **Try 1 Advantage**: Used an **Adaptive 2-Phase Budget**:
  1. *Phase 1 (Primary)*: Processed all tasks ordered cheap-first (by grid size / token count) to guarantee broad coverage.
  2. *Phase 2 (Deep Pass)*: If leftover time remained ($>20$ minutes before the 12h deadline), tasks with $<2$ valid candidate solutions were given a secondary deep search with lowered DFS pruning (`min_prob = 0.1`), new augmentation seeds, and wider search windows.
  3. **Learning**: Unequal allocation of compute time based on task difficulty recovers critical points on the leaderboard.

---

## 4. Dataset & Operational Insights

### 4.1 Dataset Properties
* **Training Challenges**: 1,000 tasks (`arc-agi_training_challenges.json`) with solutions.
* **Evaluation Challenges**: 120 tasks (`arc-agi_evaluation_challenges.json`) with solutions.
* **Test Challenges**: 240 tasks (`arc-agi_test_challenges.json`).
* **Grid Bounds**: Max $30 \times 30$, discrete integer colors $0$ to $9$.

### 4.2 Computational Environment Management
* **TensorFlow Collision**: Kaggle environments pre-install TensorFlow, which can hijack CUDA/PTX libraries and break PyTorch Triton kernel JIT compilation. Always execute `pip uninstall -y tensorflow` prior to importing Unsloth.
* **GPU Memory Hygiene**: Continuous LoRA creation and destruction leaks memory if PyTorch tensors remain in cached CUDA allocators. Mandatory pattern after every task:
  ```python
  del model, trainer, lora_model
  gc.collect()
  torch.cuda.empty_cache()
  ```

---

## 5. Future Improvement Hypotheses & Exploration Paths

1. **Hybrid Symbolic + Neural Ensembling**:
   * Integrate fast deterministic candidates from [baseline](file:///home/pika/Desktop/keggel_lb--ARC/baseline) (crop, tile, rotate, color map) as fallback candidates when TTT confidence is low.
2. **Adaptive LoRA Rank & Epochs**:
   * Scale LoRA rank and training steps dynamically based on demonstration grid size and number of examples (e.g. larger grids $\rightarrow$ smaller batch size, higher rank).
3. **Execution Verification in the Loop**:
   * If a candidate transformation rule can be inverted and evaluated on the training inputs, discard candidate predictions that violate demonstration consistency.
4. **Model Scaling**:
   * Test Qwen 2.5 7B / 14B or deep reasoning backbones with 4-bit quantization under Unsloth to evaluate whether higher base reasoning outweighs slower throughput.

---

## 6. Template for Adding Future Learnings

When adding new learnings from future runs, use the following standardized structure:

```markdown
### [YYYY-MM-DD] Experiment / Discovery Title

* **Author / Experiment Ref**: `try__X` or script name
* **Objective / Hypothesis**: What were we testing?
* **Empirical Result**: Metric delta (e.g., Score before: XX.XX -> Score after: YY.YY)
* **Key Observations**:
  * Point 1
  * Point 2
* **What Worked**: Specific changes that yielded positive gains
* **What Failed / Regressed**: Traps, slower throughput, or degraded scores
* **Action Item / Rule**: Rule to follow in subsequent implementations
```

### [2026-09-02 10:16:56] Notebook Release: arc_prize_2026_best_solution.ipynb
* **Run Ref**: `Notebook Release: arc_prize_2026_best_solution.ipynb`
* **Status / Metric**: READY FOR SUBMISSION
* **Key Observations**:
  * Created end-to-end hybrid competition notebook in my_notebook/
  * Integrated exact Symbolic Pre-Pass with Adaptive 2-Phase LoRA TTT
  * Enforced 10-minute Kaggle wall-clock safety buffer and fail-closed schema assertions
* **Execution Details**:
  * **Pipeline**: Symbolic + Unsloth Qwen TTT + DFS + score_kgmon

---
