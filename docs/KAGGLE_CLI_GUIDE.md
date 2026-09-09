# Kaggle CLI Quick Reference & Production Guide

A practical, cheat-sheet-style guide for using the **Kaggle CLI** (`v2.2+`) for competitions, notebook kernel management, dataset sync, and submission workflows.

---

## 1. Setup & Authentication

### 1.1 API Token Setup
1. Go to your Kaggle Account Settings: [https://www.kaggle.com/settings](https://www.kaggle.com/settings)
2. Scroll to the **API** section and click **Create New Token**.
3. Download `kaggle.json` and move it to `~/.kaggle/kaggle.json`:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

### 1.2 Verify Connection
```bash
kaggle competitions list
```

---

## 2. Competition Workflows (`kaggle competitions`)

### 2.1 View Submissions & Public Leaderboard Scores
Lists your submissions, status (`SubmissionStatus.COMPLETE`), dates, and public/private scores:
```bash
kaggle competitions submissions -c arc-prize-2026-arc-agi-2
```

### 2.2 Download Competition Data
```bash
# Download all files into a specific folder
kaggle competitions download -c arc-prize-2026-arc-agi-2 -p ./DATA/

# Unzip all downloaded files
unzip ./DATA/arc-prize-2026-arc-agi-2.zip -d ./DATA/
```

### 2.3 List Competition Files
```bash
kaggle competitions files -c arc-prize-2026-arc-agi-2
```

### 2.4 Submit a Direct File (CSV / JSON)
*Note: For code-competition rerun tracks like ARC Prize, submissions are made from notebook output commits via the Kaggle Web UI or through notebook submissions.*
```bash
kaggle competitions submit -c <competition-name> -f submission.csv -m "Baseline model v1"
```

---

## 3. Kernel / Notebook Management (`kaggle kernels`)

This is the primary tool for pushing local notebooks to Kaggle for remote execution on 4x NVIDIA L4 GPUs.

### 3.1 The `kernel-metadata.json` File
Every folder you push must contain a `kernel-metadata.json` file alongside your notebook:
```json
{
  "id": "soukeaizenz/arc2-hybrid-v7",
  "title": "arc2-hybrid-v7",
  "code_file": "arc2-hybrid-v7.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": false,
  "kernel_sources": [
    "soukeaizenz/pip-install-unsloth-flash-patch-fork"
  ],
  "competition_sources": [
    "arc-prize-2026-arc-agi-2"
  ],
  "model_sources": [
    "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"
  ],
  "machine_shape": "NvidiaL4"
}
```

### 3.2 Push / Update Kernel
Uploads your local notebook and metadata to Kaggle, triggering a remote execution (Save & Run All):
```bash
# Push directory containing kernel-metadata.json and the .ipynb file
kaggle kernels push -p /home/arch/ipynb/duplicate/submission/pre-sub-v7/
```

### 3.3 Check Kernel Execution Status
Checks if your notebook is `Queued`, `Running`, `Complete`, or `Error`:
```bash
kaggle kernels status soukeaizenz/arc2-hybrid-v7
```

### 3.4 Download Kernel Output Files & Logs
Downloads all artifacts produced by `/kaggle/working/` (including `submission.json` and logs):
```bash
kaggle kernels output soukeaizenz/arc2-hybrid-v7 -p ./output_dir/
```

### 3.5 Pull an Existing Kernel from Kaggle
Downloads the latest code and metadata of a notebook from Kaggle to your local disk:
```bash
kaggle kernels pull soukeaizenz/arc2-hybrid-v7 -p ./local_dir/ -m
```

---

## 4. Dataset Operations (`kaggle datasets`)

### 4.1 Download a Public Dataset
```bash
kaggle datasets download -d sorokin/qwen3_4b_grids15_sft139 -p ./models/ --unzip
```

### 4.2 Initialize and Create a New Dataset
1. Initialize metadata template:
   ```bash
   kaggle datasets init -p ./my_dataset_folder/
   ```
2. Edit `dataset-metadata.json` (`title`, `id`).
3. Upload and create:
   ```bash
   kaggle datasets create -p ./my_dataset_folder/ -r zip
   ```

### 4.3 Upload a New Version of an Existing Dataset
```bash
kaggle datasets version -p ./my_dataset_folder/ -m "Updated weights and artifacts" -r zip
```

---

## 5. ARC Competition Execution Lifecycle Tips

1. **Commit Mode vs. Rerun Mode**:
   - When you execute `kaggle kernels push`, Kaggle starts a **commit run** where `KAGGLE_IS_COMPETITION_RERUN` is **False**.
   - In Sub-7, commit mode activates the **Fast-Path** (~2 minutes), completing in <3 min and using **0 personal GPU quota**.
2. **Submitting to the Competition**:
   - Once the kernel status is `KernelWorkerStatus.COMPLETE`, visit the notebook page on Kaggle and click **Submit to Competition**.
   - Kaggle will launch the official scoring run on dedicated competition compute (`RERUN=1`, 12-hour limit, 0 personal quota consumed).
3. **Tracking Competition Status**:
   - Run `kaggle competitions submissions -c arc-prize-2026-arc-agi-2` every ~30-60 minutes to monitor your score progression and final leaderboard score.
