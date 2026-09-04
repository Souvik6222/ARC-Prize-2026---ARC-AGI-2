import json
import os

def create_notebook():
    cells = []

    # Cell 0: Markdown
    cells.append({
        "cell_type": "markdown",
        "id": "intro_v3",
        "metadata": {},
        "source": [
            "# ARC2 Hybrid v3 — Fast Symbolic Pre-Pass + Robust LoRA TTT + Diverse Attempt Ensembling\n",
            "\n",
            "This is the **V3 architecture** for the ARC Prize 2026 (ARC-AGI-2), building upon the LB 32.22% NVARC baseline.\n",
            "\n",
            "### Key V3 Improvements:\n",
            "1. **Fast Symbolic Pre-Pass Engine (`arc_symbolic.py`)**:\n",
            "   - Tests deterministic rule hypotheses (D4 symmetries, 1-to-1 color remappings, object cropping, tiling, scale, gravity) on CPU.\n",
            "   - Tasks with 100% verified exact rules across training pairs are solved instantly, saving huge GPU time for hard puzzles.\n",
            "2. **Robust LoRA TTT Training Engine (`arc_solver.py`)**:\n",
            "   - Uses native `dataset_text_field='text'` with cross-entropy loss calculation, ensuring the LoRA adapter learns demonstration patterns.\n",
            "   - Loss-aware early stopping saves 30-50% TTT time on easy puzzles.\n",
            "3. **Safe DFS Tree Search (`turbo_dfs`)**:\n",
            "   - Uses immutable token list concatenation, eliminating token sequence corruption across recursive beam branches.\n",
            "   - Explicit live logging reports every saved shard (`[Rank X] saved Y shards for ...`).\n",
            "4. **Structural Invariants & Diverse Attempts (`arc_decoder.py`)**:\n",
            "   - Candidate scoring combines consensus voting, augmented NLL, and symbolic bonus.\n",
            "   - `attempt_1`: Best overall consensus prediction.\n",
            "   - `attempt_2`: Best distinct orthogonal candidate.\n",
            "   - Unsolved tasks fall back to input identity copy.\n"
        ]
    })

    # Cell 1: Global Budget Setup
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "setup_budget",
        "metadata": {},
        "outputs": [],
        "source": [
            "# ---------------------------------------------------------------------------\n",
            "# Global wall-clock budget. Rerun (hidden test): 12h minus a 10 min write buffer.\n",
            "# Interactive/commit without hidden test: fast 55-min debug budget on 4 tasks.\n",
            "# ---------------------------------------------------------------------------\n",
            "import os, time, json, glob\n",
            "RERUN = bool(os.getenv(\"KAGGLE_IS_COMPETITION_RERUN\"))\n",
            "T0 = time.time()\n",
            "\n",
            "def has_test_challenges():\n",
            "    candidates = [\n",
            "        \"/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json\",\n",
            "        \"/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json\",\n",
            "        \"/kaggle/input/arc-prize-2026/arc-agi_test_challenges.json\",\n",
            "        \"/kaggle/input/arc-agi_test_challenges.json\",\n",
            "    ]\n",
            "    return any(os.path.exists(c) for c in candidates) or bool(glob.glob(\"/kaggle/input/**/arc*test_challenges.json\", recursive=True))\n",
            "\n",
            "if RERUN:\n",
            "    global_end_time = T0 + 12 * 3600 - 600\n",
            "elif has_test_challenges():\n",
            "    global_end_time = T0 + 11.5 * 3600 - 600\n",
            "else:\n",
            "    global_end_time = T0 + 55 * 60  # 55-minute interactive debug run\n",
            "print(f\"rerun={RERUN} budget={(global_end_time-T0)/3600:.2f}h cutoff={time.ctime(global_end_time)}\")\n"
        ]
    })

    # Cell 2: Environment Setup & sys.path sanitization
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "setup_env",
        "metadata": {},
        "outputs": [],
        "source": [
            "!pip uninstall -y tensorflow torchao\n",
            "import sys, os, glob, site\n",
            "# 1. Put site-packages at the front so real PyTorch is loaded\n",
            "for sp in reversed(site.getsitepackages()):\n",
            "    if sp in sys.path:\n",
            "        sys.path.remove(sp)\n",
            "    sys.path.insert(0, sp)\n",
            "\n",
            "# 2. Purge legacy utility script shadow folders that contain incompatible Python 3.11 numpy binaries\n",
            "sys.path = [p for p in sys.path if 'pip_install_unsloth_flash_patch' not in p and 'usr/lib/notebooks' not in p]\n",
            "if 'PYTHONPATH' in os.environ:\n",
            "    os.environ['PYTHONPATH'] = ':'.join([p for p in os.environ['PYTHONPATH'].split(':') if 'pip_install_unsloth_flash_patch' not in p and 'usr/lib' not in p])\n",
            "\n",
            "if 'torchao' in sys.modules:\n",
            "    del sys.modules['torchao']\n",
            "\n",
            "import torch\n",
            "print(sys.version.split()[0], \"torch\", torch.__version__, \"gpus\", torch.cuda.device_count(),\n",
            "      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])\n",
            "print(\"models:\", glob.glob(\"/kaggle/input/models/*/*\"), glob.glob(\"/kaggle/input/qwen*\"))\n",
            "print(\"utility scripts:\", [p for p in sys.path if \"/kaggle/usr/lib\" in p][:5])\n",
            "os.makedirs(\"/kaggle/working/logs\", exist_ok=True)\n",
            "os.makedirs(\"/kaggle/working/symbolic_outputs\", exist_ok=True)\n",
            "os.makedirs(\"/kaggle/inference_outputs\", exist_ok=True)\n",
            "os.makedirs(\"/kaggle/inference_outputs_deep\", exist_ok=True)\n"
        ]
    })

    # Read modular scripts from directory
    with open("arc_invariants.py") as f:
        invariants_code = f.read()
    with open("arc_symbolic.py") as f:
        symbolic_code = f.read()
    with open("arc_loader.py") as f:
        loader_code = f.read()
    with open("arc_decoder.py") as f:
        decoder_code = f.read()
    with open("arc_solver.py") as f:
        solver_code = f.read()
    with open("starter.py") as f:
        starter_code = f.read()

    # Cell 3: arc_invariants.py
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "write_invariants",
        "metadata": {},
        "outputs": [],
        "source": ["%%writefile arc_invariants.py\n" + invariants_code]
    })

    # Cell 4: arc_symbolic.py
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "write_symbolic",
        "metadata": {},
        "outputs": [],
        "source": ["%%writefile arc_symbolic.py\n" + symbolic_code]
    })

    # Cell 5: arc_loader.py
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "write_loader",
        "metadata": {},
        "outputs": [],
        "source": ["%%writefile arc_loader.py\n" + loader_code]
    })

    # Cell 6: arc_decoder.py
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "write_decoder",
        "metadata": {},
        "outputs": [],
        "source": ["%%writefile arc_decoder.py\n" + decoder_code]
    })

    # Cell 7: arc_solver.py
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "write_solver",
        "metadata": {},
        "outputs": [],
        "source": ["%%writefile arc_solver.py\n" + solver_code]
    })

    # Cell 8: starter.py
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "write_starter",
        "metadata": {},
        "outputs": [],
        "source": ["%%writefile starter.py\n" + starter_code]
    })

    # Cell 9: Phase 1 Primary Pass
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "phase1_exec",
        "metadata": {},
        "outputs": [],
        "source": [
            "# ---------------------------------------------------------------------------\n",
            "# Phase 1 — Primary pass: Fast Symbolic Pre-Pass on CPU + Multi-GPU LoRA TTT\n",
            "# with cheap-first task scheduling and early stopping.\n",
            "# ---------------------------------------------------------------------------\n",
            "import subprocess, sys, time, os\n",
            "os.environ.update({\n",
            "    \"UNSLOTH_DISABLE_STATISTICS\": \"1\",\n",
            "    \"TRITON_PTXAS_PATH\": \"/usr/local/cuda/bin/ptxas\",\n",
            "    \"OMP_NUM_THREADS\": \"12\",\n",
            "    \"PYTHONHASHSEED\": \"0\",\n",
            "    \"ARC_OUT_DIR\": \"/kaggle/inference_outputs\",\n",
            "    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",\n",
            "    \"PYTORCH_ALLOC_CONF\": \"expandable_segments:True\",\n",
            "    \"ARC_LORA_R\": \"64\",\n",
            "    \"ARC_GRAD_CKPT\": \"1\",\n",
            "})\n",
            "phase1_start = time.time()\n",
            "rc = subprocess.call([sys.executable, \"starter.py\", \"--end-time\", f\"{global_end_time}\", \"--order\", \"cheap\"])\n",
            "print(f\"phase-1 rc={rc} took {(time.time()-phase1_start)/60:.1f} min; remaining {(global_end_time-time.time())/60:.1f} min\")\n"
        ]
    })

    # Cell 10: Phase 2 Adaptive Catch-Up & Deep Pass
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "phase2_exec",
        "metadata": {},
        "outputs": [],
        "source": [
            "# ---------------------------------------------------------------------------\n",
            "# Phase 2 — Adaptive deep pass on starved/uncertain tasks if time remains (>20 min)\n",
            "# ---------------------------------------------------------------------------\n",
            "import os, sys, json, time, subprocess\n",
            "from arc_loader import ArcDataset\n",
            "from arc_decoder import ArcDecoder\n",
            "\n",
            "def remaining():\n",
            "    return global_end_time - time.time()\n",
            "\n",
            "def resolve_challenges_path(rerun=False):\n",
            "    fname = 'arc-agi_test_challenges.json' if rerun else 'arc-agi_evaluation_challenges.json'\n",
            "    alt_fname = 'arc_agi_test_challenges.json' if rerun else 'arc_agi_evaluation_challenges.json'\n",
            "    candidates = [\n",
            "        f'/kaggle/input/competitions/arc-prize-2026-arc-agi-2/{fname}',\n",
            "        f'/kaggle/input/arc-prize-2026-arc-agi-2/{fname}',\n",
            "        f'/kaggle/input/arc-prize-2026/{fname}',\n",
            "        f'/kaggle/input/{fname}',\n",
            "        f'/kaggle/input/competitions/arc-prize-2026-arc-agi-2/{alt_fname}',\n",
            "        f'/kaggle/input/arc-prize-2026-arc-agi-2/{alt_fname}',\n",
            "        f'/kaggle/input/arc-prize-2026/{alt_fname}',\n",
            "    ]\n",
            "    for c in candidates:\n",
            "        if os.path.exists(c):\n",
            "            return c\n",
            "    import glob\n",
            "    for pattern in [f'/kaggle/input/**/{fname}', f'/kaggle/input/**/{alt_fname}']:\n",
            "        matches = glob.glob(pattern, recursive=True)\n",
            "        if matches:\n",
            "            return matches[0]\n",
            "    return f'/kaggle/input/competitions/arc-prize-2026-arc-agi-2/{fname}'\n",
            "\n",
            "test_file = resolve_challenges_path(rerun=True)\n",
            "test_path = test_file if (RERUN or os.path.exists(test_file)) else resolve_challenges_path(rerun=False)\n",
            "\n",
            "data = ArcDataset.from_file(test_path)\n",
            "keys_in_scope = data.keys if (RERUN or os.path.exists(test_file)) else [k for k in data.keys if k in os.getenv(\"ARC_DEBUG_KEYS\", \"0934a4d8,36a08778,981571dc,aa4ec2a5\").split(\",\")]\n",
            "\n",
            "dec = ArcDecoder(data.split_multi_replies(), n_guesses=2)\n",
            "dec.load_decoded_results(\"/kaggle/inference_outputs\")\n",
            "dec.load_decoded_results(\"/kaggle/working/symbolic_outputs\", run_name=\".sym\")\n",
            "stats = dec.candidate_stats()\n",
            "\n",
            "unprocessed, starved = [], {}\n",
            "for k in keys_in_scope:\n",
            "    n_out = len(data.queries[k][\"test\"])\n",
            "    outs = [f\"{k}_{i}\" for i in range(n_out)]\n",
            "    if not any(o in stats for o in outs):\n",
            "        unprocessed.append(k)\n",
            "        continue\n",
            "    # Skip tasks that already have an exact verified symbolic rule solution\n",
            "    has_sym_exact = all(any(s.get(\"is_symbolic_exact\") for s in dec.decoded_results.get(o, {}).values()) for o in outs)\n",
            "    if has_sym_exact:\n",
            "        continue\n",
            "    n_starved = sum(1 for o in outs if stats.get(o, {\"unique\": 0})[\"unique\"] < 2)\n",
            "    if n_starved:\n",
            "        starved[k] = n_starved\n",
            "\n",
            "print(f\"[phase-2] unprocessed={len(unprocessed)} starved_tasks={len(starved)} remaining={remaining()/60:.1f} min\")\n",
            "json.dump({\"unprocessed\": unprocessed, \"starved\": starved}, open(\"/kaggle/working/phase2_plan.json\", \"w\"))\n",
            "\n",
            "base_env = dict(os.environ, UNSLOTH_DISABLE_STATISTICS=\"1\", TRITON_PTXAS_PATH=\"/usr/local/cuda/bin/ptxas\",\n",
            "                OMP_NUM_THREADS=\"12\", PYTHONHASHSEED=\"0\", PYTORCH_CUDA_ALLOC_CONF=\"expandable_segments:True\", PYTORCH_ALLOC_CONF=\"expandable_segments:True\", ARC_LORA_R=\"64\", ARC_GRAD_CKPT=\"1\")\n",
            "\n",
            "def run_phase(name, keys, env_overrides, out_dir):\n",
            "    if not keys or remaining() < 20 * 60:\n",
            "        print(f\"[phase-2:{name}] skipped (keys={len(keys)}, remaining={remaining()/60:.1f} min)\")\n",
            "        return\n",
            "    keys_file = f\"/kaggle/working/keys_{name}.json\"\n",
            "    json.dump(keys, open(keys_file, \"w\"))\n",
            "    env = dict(base_env, ARC_OUT_DIR=out_dir, **{k: str(v) for k, v in env_overrides.items()})\n",
            "    t = time.time()\n",
            "    rc = subprocess.call([sys.executable, \"starter.py\", \"--end-time\", f\"{global_end_time}\", \"--keys-file\", keys_file, \"--order\", \"file\", \"--skip-symbolic\"], env=env)\n",
            "    print(f\"[phase-2:{name}] rc={rc} keys={len(keys)} took {(time.time()-t)/60:.1f} min; remaining {remaining()/60:.1f} min\")\n",
            "\n",
            "# (a) Catch-up with primary configuration\n",
            "from starter import estimated_work\n",
            "unprocessed = sorted(unprocessed, key=lambda k: estimated_work(data.queries[k]))\n",
            "run_phase(\"catchup\", unprocessed, {}, \"/kaggle/inference_outputs\")\n",
            "\n",
            "# (b) Deep pass on starved outputs: most starved outputs first, then cheapest\n",
            "deep_keys = sorted(starved, key=lambda k: (-starved[k], estimated_work(data.queries[k])))\n",
            "if deep_keys and remaining() >= 20 * 60:\n",
            "    per_task = max(600.0, min(2400.0, (remaining() - 300) * 4.0 / len(deep_keys)))\n",
            "    deep_cfg = dict(ARC_LORA_SEED=137, ARC_TRAIN_AUG_SEED=17, ARC_EVAL_AUG_SEED=29, ARC_N_EVAL_AUG=3,\n",
            "                    ARC_MIN_PROB=0.1, ARC_DFS_WINDOW=600, ARC_TASK_CAP=int(per_task), ARC_SCORE_SEED_OFFSET=7)\n",
            "    print(f\"[phase-2:deep] per-task cap {per_task:.0f}s, config {deep_cfg}\")\n",
            "    run_phase(\"deep\", deep_keys, deep_cfg, \"/kaggle/inference_outputs_deep\")\n"
        ]
    })

    # Cell 11: Phase 3 Selection, Diverse Attempts & Submission Assembly
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "phase3_assembly",
        "metadata": {},
        "outputs": [],
        "source": [
            "# ---------------------------------------------------------------------------\n",
            "# Phase 3 — Final Selection, Diverse Attempt Generation, Schema Validation\n",
            "# ---------------------------------------------------------------------------\n",
            "import os, json, hashlib\n",
            "import numpy as np\n",
            "from arc_loader import ArcDataset\n",
            "from arc_decoder import ArcDecoder, score_v2\n",
            "\n",
            "def resolve_file(filenames):\n",
            "    candidates = []\n",
            "    for fname in filenames:\n",
            "        candidates.extend([\n",
            "            f'/kaggle/input/competitions/arc-prize-2026-arc-agi-2/{fname}',\n",
            "            f'/kaggle/input/arc-prize-2026-arc-agi-2/{fname}',\n",
            "            f'/kaggle/input/arc-prize-2026/{fname}',\n",
            "            f'/kaggle/input/{fname}',\n",
            "        ])\n",
            "    for c in candidates:\n",
            "        if os.path.exists(c):\n",
            "            return c\n",
            "    import glob\n",
            "    for fname in filenames:\n",
            "        matches = glob.glob(f'/kaggle/input/**/{fname}', recursive=True)\n",
            "        if matches:\n",
            "            return matches[0]\n",
            "    return candidates[0]\n",
            "\n",
            "test_file = resolve_file(['arc-agi_test_challenges.json', 'arc_agi_test_challenges.json'])\n",
            "test_path = test_file if (RERUN or os.path.exists(test_file)) else resolve_file(['arc-agi_evaluation_challenges.json', 'arc_agi_evaluation_challenges.json'])\n",
            "\n",
            "data = ArcDataset.from_file(test_path)\n",
            "sol_file = resolve_file(['arc-agi_evaluation_solutions.json', 'arc_agi_evaluation_solutions.json'])\n",
            "if not RERUN and os.path.exists(sol_file):\n",
            "    data = data.load_replies(sol_file)\n",
            "\n",
            "decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)\n",
            "decoder.load_decoded_results(\"/kaggle/working/symbolic_outputs\", run_name=\".sym\")\n",
            "decoder.load_decoded_results(\"/kaggle/inference_outputs\")\n",
            "decoder.load_decoded_results(\"/kaggle/inference_outputs_deep\", run_name=\".deep\")\n",
            "\n",
            "# Generate orthogonal/diverse attempts for each test query\n",
            "diverse_attempts = decoder.get_diverse_attempts(selection_algorithm=score_v2)\n",
            "\n",
            "submission = {k: [{f'attempt_{i+1}': [[0]] for i in range(2)} for _ in range(len(data.queries[k]['test']))] for k in data.keys}\n",
            "\n",
            "n_fallback = 0\n",
            "for k in data.keys:\n",
            "    for i, t in enumerate(data.queries[k][\"test\"]):\n",
            "        subkey = f\"{k}_{i}\"\n",
            "        target_entry = submission[k][i]\n",
            "        if subkey in diverse_attempts:\n",
            "            att1, att2 = diverse_attempts[subkey]\n",
            "            target_entry[\"attempt_1\"] = att1.tolist()\n",
            "            target_entry[\"attempt_2\"] = att2.tolist()\n",
            "        else:\n",
            "            target_entry[\"attempt_1\"] = [[int(x) for x in row] for row in t[\"input\"]]\n",
            "            target_entry[\"attempt_2\"] = [[0]]\n",
            "            n_fallback += 1\n",
            "\n",
            "        # Schema hardening\n",
            "        for a in (\"attempt_1\", \"attempt_2\"):\n",
            "            g = target_entry.get(a)\n",
            "            ok = isinstance(g, list) and len(g) > 0 and len(g) <= 30 and all(isinstance(r, list) and len(r) == len(g[0]) and 0 < len(r) <= 30 for r in g)\n",
            "            if not ok:\n",
            "                if a == \"attempt_1\":\n",
            "                    target_entry[a] = [[int(x) for x in row] for row in t[\"input\"]]\n",
            "                else:\n",
            "                    target_entry[a] = [[0]]\n",
            "            target_entry[a] = [[int(x) for x in row] for row in target_entry[a]]\n",
            "\n",
            "        if target_entry[\"attempt_2\"] == target_entry[\"attempt_1\"]:\n",
            "            target_entry[\"attempt_2\"] = [[0]]\n",
            "\n",
            "with open(\"/kaggle/working/submission.json\", \"w\") as f:\n",
            "    json.dump(submission, f)\n",
            "\n",
            "print(f\"*** wrote submission.json: {len(submission)} tasks, {sum(len(v) for v in submission.values())} outputs, {n_fallback} fallbacks, \"\n",
            "      f\"sha256={hashlib.sha256(open('/kaggle/working/submission.json','rb').read()).hexdigest()[:12]}\")\n",
            "\n",
            "# Integrity verification\n",
            "chk = json.load(open(\"/kaggle/working/submission.json\"))\n",
            "assert set(chk) == set(data.keys), \"missing task ids\"\n",
            "for k in data.keys:\n",
            "    assert len(chk[k]) == len(data.queries[k][\"test\"]), f\"wrong #outputs for {k}\"\n",
            "    for e in chk[k]:\n",
            "        assert \"attempt_1\" in e and \"attempt_2\" in e\n",
            "        assert isinstance(e[\"attempt_1\"], list) and isinstance(e[\"attempt_2\"], list)\n",
            "print(\"*** submission schema verified OK!\")\n",
            "\n",
            "if not RERUN and hasattr(data, \"replies\") and data.replies:\n",
            "    val_score = data.validate_submission(chk)\n",
            "    print(f\"*** Validation score: {val_score:.2f} of {len(data.keys)} tasks\")\n"
        ]
    })

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.13"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open("arc2-hybrid-v3.ipynb", "w") as f:
        json.dump(nb, f, indent=2)

    print("Created arc2-hybrid-v3.ipynb successfully!")

if __name__ == "__main__":
    create_notebook()
