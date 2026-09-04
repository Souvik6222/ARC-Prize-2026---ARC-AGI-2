"""
arc_solver.py - Test-Time Training (LoRA) & Constrained Turbo DFS Tree Search Engine for ARC-AGI-2
"""
import sys
import os
import glob
# Fix6: reduce fragmentation — must be set before CUDA context init
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

sys.setrecursionlimit(5000)

import site
# Prioritize real system site-packages so real PyTorch is loaded
for sp in reversed(site.getsitepackages()):
    if sp in sys.path:
        sys.path.remove(sp)
    sys.path.insert(0, sp)

# Remove any broken torchao from sys.modules
if "torchao" in sys.modules:
    del sys.modules["torchao"]

# Patch peft to safely treat torchao as unavailable
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
except Exception:
    pass
try:
    import peft.tuners.lora.torchao
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
except Exception:
    pass

# Move any utility script shadow folders to the end of sys.path
for p in list(sys.path):
    if "pip_install_unsloth_flash_patch" in p or "usr/lib/notebooks" in p:
        sys.path.remove(p)
        sys.path.append(p)

# Discover unsloth from Kaggle utility scripts or input datasets and append to sys.path
_unsloth_search_paths = (
    glob.glob("/kaggle/usr/lib/notebooks/*/pip_install_unsloth*") +
    glob.glob("/kaggle/usr/lib/notebooks/*/*/pip_install_unsloth*") +
    glob.glob("/kaggle/usr/lib/**", recursive=True) +
    glob.glob("/kaggle/input/**/unsloth", recursive=True)
)
for p in _unsloth_search_paths:
    if os.path.isdir(p):
        if os.path.isfile(os.path.join(p, "unsloth", "__init__.py")):
            if p not in sys.path:
                sys.path.append(p)
        elif os.path.basename(p) == "unsloth" and os.path.isfile(os.path.join(p, "__init__.py")):
            parent = os.path.dirname(p)
            if parent not in sys.path:
                sys.path.append(parent)

try:
    from unsloth import FastLanguageModel, UnslothTrainingArguments, UnslothTrainer
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model
    UnslothTrainingArguments = TrainingArguments
    UnslothTrainer = Trainer

from arc_loader import ArcDataset, QwenFormatter, is_valid_solution
from arc_decoder import hashable
from arc_invariants import is_valid_grid

import gc
import io
import time
import zlib
import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from collections import defaultdict
from typing import Any, Union, List, Dict, Tuple, Optional
from transformers import DataCollatorForLanguageModeling, TrainerCallback
import logging
from contextlib import redirect_stdout, redirect_stderr
from peft import get_peft_model_state_dict, set_peft_model_state_dict
import bz2
import pickle
import traceback

logging.disable(logging.WARNING)

def _env(name, default, cast):
    v = os.getenv(name)
    return cast(v) if v not in (None, "") else default

CFG = dict(
    model_path      = _env("ARC_MODEL_PATH", "", str),
    out_dir         = _env("ARC_OUT_DIR", "/kaggle/inference_outputs", str),
    lora_seed       = _env("ARC_LORA_SEED", 42, int),
    train_aug_seed  = _env("ARC_TRAIN_AUG_SEED", 1, int),
    n_train_aug     = _env("ARC_N_TRAIN_AUG", 16, int),     # x8 geometries = 128 sequences
    num_epochs      = _env("ARC_EPOCHS", 1, int),
    learning_rate   = _env("ARC_LR", 5e-5, float),
    eval_aug_seed   = _env("ARC_EVAL_AUG_SEED", 2, int),
    n_eval_aug      = _env("ARC_N_EVAL_AUG", 2, int),       # x8 geometries = 16 decoded views
    min_prob        = _env("ARC_MIN_PROB", 0.2, float),     # DFS cumulative prob threshold
    dfs_window      = _env("ARC_DFS_WINDOW", 540.0, float), # seconds per DFS call
    task_cap        = _env("ARC_TASK_CAP", 1200.0, float),  # seconds per task (decode phase)
    score_seed_off  = _env("ARC_SCORE_SEED_OFFSET", 0, int),
    decode_batch    = _env("ARC_DECODE_BATCH", 4, int),
    early_stop_loss = _env("ARC_EARLY_STOP_LOSS", 5e-4, float), # TTT loss threshold for early stop
)

ARC_VOCAB = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
    "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "Ċ": 10, "<|im_end|>": 15,
}

ARC_TOKENS = list(ARC_VOCAB.values())
USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
PAD_ID = 13
EOS_ID = 15


def resolve_model_dir():
    if CFG["model_path"] and os.path.isdir(CFG["model_path"]):
        return CFG["model_path"]
    candidates = [
        "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
        "/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
        "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1",
        "/kaggle/input/arc-qwen-model",
        "/kaggle/input/qwen-models",
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "config.json")):
            return c
    for c in glob.glob("/kaggle/input/**/config.json", recursive=True):
        d = os.path.dirname(c)
        if os.path.isfile(os.path.join(d, "tokenizer.json")) or os.path.isfile(os.path.join(d, "tokenizer_config.json")):
            return d
    return "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"


def stable_seed(key, offset=0):
    return (zlib.crc32(key.encode("utf-8")) + offset) % (1024 ** 2)


def make_training_args(**kwargs):
    if HAS_UNSLOTH:
        return UnslothTrainingArguments(**kwargs)
    import inspect
    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return TrainingArguments(**kwargs)
    cleaned = {k: v for k, v in kwargs.items() if k in params}
    return TrainingArguments(**cleaned)


class EarlyStoppingOnLossCallback(TrainerCallback):
    """Stops TTT fine-tuning early if demonstration loss has converged near zero."""
    def __init__(self, threshold=5e-4):
        self.threshold = threshold

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            if logs["loss"] < self.threshold:
                control.should_training_stop = True


class UnslothFixedTrainer(UnslothTrainer):

    def __init__(self, *args, **kwargs):
        import inspect
        sig = inspect.signature(UnslothTrainer.__init__)
        params = sig.parameters
        cleaned = {}
        for k, v in kwargs.items():
            if k in params:
                cleaned[k] = v
            elif k == "tokenizer" and "processing_class" in params:
                cleaned["processing_class"] = v
            elif k == "processing_class" and "tokenizer" in params:
                cleaned["tokenizer"] = v
        super().__init__(*args, **cleaned)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        import torch.nn as nn
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if getattr(self.args, "past_index", -1) >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "_get_name") and "unsloth" in unwrapped_model._get_name().lower():
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(outputs.logits.view(-1, outputs.logits.shape[-1]), labels.view(-1))
            else:
                loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError("The model did not return a loss from inputs.")
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if hasattr(loss, "clone"):
            loss = loss.clone()
        if self.accelerator.num_processes > 1:
            loss = loss * self.accelerator.num_processes
        return (loss, outputs) if return_outputs else loss


class QwenDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)
        for i in range(len(examples)):
            labels = batch["input_ids"][i].clone()
            labels_np = labels.detach().cpu().numpy()
            user_start_idx = np.where(labels_np == USER_TOKEN_ID)[0].tolist()
            assistant_start_idx = np.where(labels_np == ASSISTANT_TOKEN_ID)[0].tolist()
            start_idx = sorted(user_start_idx + assistant_start_idx)
            end_idx = np.where(labels_np == EOS_ID)[0]
            batch["labels"][i, :] = -100
            for j, (start, end) in enumerate(zip(start_idx, end_idx)):
                assert start < end
                if j % 2 == 1:
                    start += 2
                    end += 1
                    batch["labels"][i, start:end] = labels[start:end]
        return batch


_ARC_TOKEN_ID_CACHE = {}

def _arc_token_ids(device):
    key = str(device)
    token_ids = _ARC_TOKEN_ID_CACHE.get(key)
    if token_ids is None:
        token_ids = torch.tensor(ARC_TOKENS, dtype=torch.long, device=device)
        _ARC_TOKEN_ID_CACHE[key] = token_ids
    return token_ids


def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache, start_time, end_time, dfs_window) -> dict:
    n = logits.size(0)
    logits_f = logits.float()
    token_ids = _arc_token_ids(logits.device)
    arc_logits = logits_f.index_select(-1, token_ids)
    nll = (
        torch.as_tensor(scores, dtype=torch.float32, device=logits.device).view(n, 1)
        + torch.logsumexp(logits_f, dim=-1, keepdim=True)
        - arc_logits
    ).cpu()

    suffixes = defaultdict(list)
    candidates = dict()

    for i in range(n):
        candidates[i] = []
        for token_idx, t in enumerate(ARC_TOKENS):
            score = nll[i, token_idx].item()
            if score < max_score:
                if t == EOS_ID:
                    suffixes[i].append((score, [t]))
                elif max_new_tokens > 1:
                    candidates[i].append((score, t))

    for i in range(n):
        candidates[i] = sorted(candidates[i], key=lambda x: x[0])

    while time.time() - start_time < dfs_window and time.time() < end_time:
        batch_tokens = []
        batch_scores = []
        num_alive_beams = 0

        for i in range(n):
            if len(candidates[i]) == 0:
                batch_tokens.append(PAD_ID)
                batch_scores.append(1000)
            else:
                score, t = candidates[i].pop(0)
                batch_tokens.append(t)
                batch_scores.append(score)
                num_alive_beams += 1

        if num_alive_beams == 0:
            break

        outputs = model(
            input_ids=torch.tensor(batch_tokens, device=model.device, dtype=torch.long).view(-1, 1),
            position_ids=torch.full((n, 1), pos, device=model.device),
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )

        next_suffixes = turbo_dfs(
            model,
            logits=outputs.logits[:, -1],
            max_new_tokens=max_new_tokens - 1,
            max_score=max_score,
            scores=batch_scores,
            pos=pos + 1,
            cache=outputs.past_key_values,
            start_time=start_time,
            end_time=end_time,
            dfs_window=dfs_window,
        )

        for batch_id, beams in next_suffixes.items():
            for score, suffix_tokens in beams:
                suffixes[batch_id].append((score, [batch_tokens[batch_id]] + suffix_tokens))
        # Fix4: free per-iteration KV and logits to avoid recursive cache blowup
        try:
            del outputs, next_suffixes, batch_tokens, batch_scores
        except Exception:
            pass

    # free temporary NLL tensors
    try:
        del logits_f, arc_logits, nll, token_ids
    except Exception:
        pass
    return suffixes


@torch.no_grad()
def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time, dfs_window):
    input_ids = torch.tensor(prefix_tokens, device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    # cache reference kept for turbo_dfs, free logits after
    _logits_slice = outputs.logits[:, -1].detach()
    _past = outputs.past_key_values
    # free full outputs logits early to save memory
    try:
        del outputs
    except Exception:
        pass
    suffixes = turbo_dfs(
        model,
        logits=_logits_slice,
        max_new_tokens=max_new_tokens,
        max_score=max_score,
        scores=[0.0] * input_ids.size(0),
        pos=input_ids.size(1),
        cache=_past,
        start_time=time.time(),
        end_time=end_time,
        dfs_window=dfs_window,
    )
    try:
        del _logits_slice, _past, input_ids
    except Exception:
        pass
    result = []
    for batch_id, beams in suffixes.items():
        sorted_beams = sorted(beams, key=lambda x: x[0])
        result.append((batch_id, sorted_beams))
    try:
        del suffixes
    except Exception:
        pass
    return result


@torch.no_grad()
def calc_scores(queries, answers, tokenizer, model):
    batch_query_tokens = []
    batch_answer_tokens = []
    batch_tokens = []
    batch_lengths = []
    for query, answer in zip(queries, answers):
        query_tokens = tokenizer.encode(query)
        answer_tokens = tokenizer.encode(answer)
        tokens = query_tokens + answer_tokens
        batch_query_tokens.append(query_tokens)
        batch_answer_tokens.append(answer_tokens)
        batch_tokens.append(tokens)
        batch_lengths.append(len(tokens))
    max_len = max(batch_lengths)
    padded_tokens = []
    for tokens in batch_tokens:
        padded = tokens + [PAD_ID] * (max_len - len(tokens))
        padded_tokens.append(padded)
    input_ids = torch.tensor(padded_tokens, device=model.device, dtype=torch.long)

    outputs = model(input_ids=input_ids, return_dict=True, use_cache=False)
    batch_logits = outputs.logits.float()
    # free outputs early — keep only float logits
    try:
        del outputs, padded_tokens
    except Exception:
        pass
    batch_log_norm = torch.logsumexp(batch_logits, dim=-1)
    result = []
    for row_id, (query_tokens, answer_tokens) in enumerate(zip(batch_query_tokens, batch_answer_tokens)):
        query_length = len(query_tokens)
        answer_length = len(answer_tokens)
        positions = torch.arange(
            query_length - 1,
            query_length - 1 + answer_length,
            device=model.device,
        )
        target_tokens = torch.tensor(answer_tokens, device=model.device, dtype=torch.long)
        answer_log_probs = (
            batch_logits[row_id, positions, target_tokens]
            - batch_log_norm[row_id, positions]
        )
        result.append(-answer_log_probs.sum().item())
        try:
            del positions, target_tokens, answer_log_probs
        except Exception:
            pass
    # Fix5: free large logits tensors before return
    try:
        del input_ids, batch_logits, batch_log_norm, batch_query_tokens, batch_answer_tokens, batch_tokens, batch_lengths
    except Exception:
        pass
    return result


def make_view_batches(eval_ds, n_perm, batch_size):
    test_id_to_subkeys = defaultdict(list)
    for subkey in sorted(eval_ds.keys):
        test_id = subkey.split(".")[0].split("_")[1]
        test_id_to_subkeys[test_id].append(subkey)
    groups_a = [0, 2, 1, 3]
    groups_b = [4, 6, 5, 7]
    batches = []
    for geos in (groups_a, groups_b):
        for test_id, subkeys in test_id_to_subkeys.items():
            if n_perm == 2 and batch_size == 4:
                for a, b in ((geos[0], geos[1]), (geos[2], geos[3])):
                    batches.append(subkeys[a * n_perm:(a + 1) * n_perm] + subkeys[b * n_perm:(b + 1) * n_perm])
            else:
                views = []
                for g in geos:
                    views.extend(subkeys[g * n_perm:(g + 1) * n_perm])
                for i in range(0, len(views), batch_size):
                    batches.append(views[i:i + batch_size])
    return batches


def worker(rank, queue, end_time, test_path=None):
    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    peft_params = dict(
        r=_env("ARC_LORA_R", 64, int),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=_env("ARC_GRAD_CKPT", 1, int) == 1,
        random_state=CFG["lora_seed"],
        use_rslora=True,
        loftq_config=None,
    )

    train_args = dict(
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=CFG["num_epochs"],
        warmup_steps=0,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        learning_rate=CFG["learning_rate"],
        optim="adamw_torch",
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        seed=CFG["lora_seed"],
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="steps",
        logging_steps=16,
        fp16=False,
        bf16=True,
        fsdp="",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        gradient_checkpointing=_env("ARC_GRAD_CKPT", 1, int) == 1,
        remove_unused_columns=False,
    )

    max_seq_length = 8192

    model_dir = resolve_model_dir()
    print(f"[Rank {rank}] model dir: {model_dir}")
    print(f"[Rank {rank}] config: {CFG}")

    if HAS_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            full_finetuning=False,
            load_in_4bit=False,
            local_files_only=True,
            use_gradient_checkpointing=_env("ARC_GRAD_CKPT", 1, int) == 1,
            max_seq_length=max_seq_length,
        )
        model = FastLanguageModel.get_peft_model(model, **peft_params)
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, local_files_only=True)
        peft_config = LoraConfig(
            r=peft_params.get("r", 64),
            lora_alpha=peft_params.get("lora_alpha", 32),
            target_modules=peft_params.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            lora_dropout=peft_params.get("lora_dropout", 0.0),
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        model = model.to("cuda")
        if _env("ARC_GRAD_CKPT", 1, int) == 1:
            try:
                model.gradient_checkpointing_enable()
                model.enable_input_require_grads()
            except Exception:
                pass

    for name, param in model.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)

    default_weights = get_peft_model_state_dict(model, adapter_name="default")
    # Fix2: keep LoRA snapshot on CPU to free ~1-2GB GPU across tasks
    default_weights = {k: v.clone().detach().cpu() for k, v in default_weights.items()}

    collator = QwenDataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        mlm=False,
    )

    formatter = QwenFormatter(tokenizer=tokenizer)
    max_new_tokens = formatter.max_new_tokens()
    max_score = -np.log(CFG["min_prob"])

    if test_path is None:
        candidates_test = [
            "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json",
            "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json",
            "/kaggle/input/arc-prize-2026/arc-agi_test_challenges.json",
            "/kaggle/input/arc-agi_test_challenges.json",
            "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc_agi_test_challenges.json",
            "/kaggle/input/arc-prize-2026-arc-agi-2/arc_agi_test_challenges.json",
            "/kaggle/input/arc-prize-2026/arc_agi_test_challenges.json",
        ]
        candidates_eval = [
            "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json",
            "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json",
            "/kaggle/input/arc-prize-2026/arc-agi_evaluation_challenges.json",
            "/kaggle/input/arc-agi_evaluation_challenges.json",
            "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc_agi_evaluation_challenges.json",
            "/kaggle/input/arc-prize-2026-arc-agi-2/arc_agi_evaluation_challenges.json",
            "/kaggle/input/arc-prize-2026/arc_agi_evaluation_challenges.json",
        ]
        resolved_test = next((c for c in candidates_test if os.path.exists(c)), None)
        if resolved_test is None:
            matches = glob.glob("/kaggle/input/**/arc*test_challenges.json", recursive=True)
            if matches:
                resolved_test = matches[0]

        if rerun_mode:
            test_path = resolved_test or candidates_test[0]
        else:
            resolved_eval = next((c for c in candidates_eval if os.path.exists(c)), None)
            if resolved_eval is None:
                matches = glob.glob("/kaggle/input/**/arc*evaluation_challenges.json", recursive=True)
                if matches:
                    resolved_eval = matches[0]
            test_path = resolved_eval or candidates_eval[0]

    arc_test_set = ArcDataset.from_file(test_path)
    dir_outputs = CFG["out_dir"]
    os.makedirs(dir_outputs, exist_ok=True)

    early_stop_cb = EarlyStoppingOnLossCallback(threshold=CFG["early_stop_loss"])

    while True:
        if time.time() > end_time:
            print(f"[Rank {rank}] stop!")
            break

        key = queue.get()
        if key is None:
            break

        start_time = time.time()

        try:
            torch.cuda.reset_peak_memory_stats()

            # Fix2: restore from CPU snapshot, moving to model device
            _dw_on_device = {k: v.to(model.device) for k, v in default_weights.items()}
            set_peft_model_state_dict(
                model,
                _dw_on_device,
                adapter_name="default",
            )
            del _dw_on_device

            if HAS_UNSLOTH:
                model = FastLanguageModel.for_training(model)

            puzzle_ds = arc_test_set.change_keys([key])
            train_ds = puzzle_ds.augment(n=CFG["n_train_aug"], shfl_keys=True, seed=CFG["train_aug_seed"])
            train_ds = train_ds.cut_to_len(formatter=formatter, name="text", max_len=max_seq_length)

            if HAS_UNSLOTH:
                train_data = Dataset.from_list(train_ds.as_list(formatter))
                extra_kwargs = dict(
                    dataset_text_field="text",
                    max_seq_length=max_seq_length,
                    args=UnslothTrainingArguments(**train_args),
                )
            else:
                train_items = []
                for item in train_ds.as_list(formatter):
                    tokens = tokenizer.encode(item["text"])
                    train_items.append({"input_ids": tokens})
                train_data = Dataset.from_list(train_items)
                extra_kwargs = dict(
                    args=make_training_args(**train_args),
                )

            with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
                trainer = UnslothFixedTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    data_collator=collator,
                    train_dataset=train_data,
                    callbacks=[early_stop_cb] if CFG.get("early_stop_loss", 0) > 0 else [],
                    **extra_kwargs,
                )
                stats = trainer.train()
                model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
                del trainer

            if HAS_UNSLOTH:
                model = FastLanguageModel.for_inference(model)

            try:
                del trainer  # already deleted above, ensure freed
            except Exception:
                pass
            gc.collect()
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

            memory_allocated = torch.cuda.max_memory_allocated() // 1024 ** 2
            print(f"[Rank {rank}] allocated {memory_allocated}MB for training")
            torch.cuda.reset_peak_memory_stats()
            print(f"[Rank {rank}] training stats for puzzle {key}: {stats}")
            try:
                del train_ds
            except Exception:
                pass
            gc.collect()
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

            puzzle_ds_multi = puzzle_ds.split_multi_replies()
            eval_ds = puzzle_ds_multi.augment(n=CFG["n_eval_aug"], seed=CFG["eval_aug_seed"])
            eval_ds = eval_ds.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length - max_new_tokens)

            batches = make_view_batches(eval_ds, CFG["n_eval_aug"], CFG["decode_batch"])

            with torch.inference_mode():
                known_scores = {}
                for subkeys in batches:
                    spend_time = time.time() - start_time
                    if spend_time > CFG["task_cap"] or time.time() > end_time:
                        print(f"[Rank {rank}] timeout after {spend_time:.1f}s for puzzle {key}")
                        break

                    print(f"[Rank {rank}] decoding {subkeys}")
                    tokens = []
                    for subkey in subkeys:
                        data = eval_ds.get(subkey, formatter)
                        tokens.append(tokenizer.encode(data["input"]))

                    dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, max_score, end_time, CFG["dfs_window"])

                    for subkey_id, scored_beams in dfs_result:
                        subkey = subkeys[subkey_id]
                        bk = subkey.split(".")[0]
                        decoded_result = []

                        for beam_score, tokens_ in scored_beams:
                            array = formatter.convert_tokens_to_array(tokens_)
                            if array is None:
                                continue

                            solution = puzzle_ds_multi.invert_mod(array, subkey, inv_perm=True)
                            if not is_valid_solution(solution):
                                continue

                            grid_id = (bk, tuple(map(tuple, solution)))

                            if grid_id in known_scores:
                                augmented_scores = known_scores[grid_id]
                            else:
                                aug_dataset = ArcDataset(
                                    keys=[bk],
                                    queries={bk: puzzle_ds_multi.queries.get(bk)},
                                    replies={bk: [solution.tolist()]},
                                )
                                aug_dataset = aug_dataset.augment(seed=stable_seed(bk, CFG["score_seed_off"]))
                                aug_dataset = aug_dataset.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length - max_new_tokens)
                                aug_queries = []
                                aug_answers = []
                                for augmented_sample in aug_dataset.as_list(formatter):
                                    aug_queries.append(augmented_sample["input"])
                                    aug_answers.append(augmented_sample["reply"])
                                augmented_scores1 = calc_scores(aug_queries[:4], aug_answers[:4], tokenizer, model)
                                augmented_scores2 = calc_scores(aug_queries[4:], aug_answers[4:], tokenizer, model)
                                augmented_scores = augmented_scores1 + augmented_scores2
                                known_scores[grid_id] = augmented_scores
                                # Fix5: free augmentation artifacts immediately
                                try:
                                    del aug_dataset, aug_queries, aug_answers, augmented_scores1, augmented_scores2
                                except Exception:
                                    pass

                            decoded_result.append({
                                "beam_score": beam_score,
                                "score_aug": augmented_scores,
                                "solution": solution,
                            })

                        if len(decoded_result):
                            shard_path = os.path.join(dir_outputs, subkey)
                            with bz2.BZ2File(shard_path, "w") as f:
                                pickle.dump(decoded_result, f)
                            print(f"[Rank {rank}] saved {len(decoded_result)} shards for {subkey}")
                        try:
                            del decoded_result, scored_beams
                        except Exception:
                            pass
                    # per-batch cleanup
                    try:
                        del dfs_result, tokens
                    except Exception:
                        pass
                    gc.collect()

            memory_allocated = torch.cuda.max_memory_allocated() // 1024 ** 2
            print(f"[Rank {rank}] allocated {memory_allocated}MB for inference")
            # --- Fix1: symmetric hygiene — free inference artifacts before next task ---
            try:
                del puzzle_ds
            except Exception:
                pass
            try:
                del train_ds
            except Exception:
                pass
            try:
                del puzzle_ds_multi
            except Exception:
                pass
            try:
                del eval_ds
            except Exception:
                pass
            try:
                del batches
            except Exception:
                pass
            try:
                del known_scores
            except Exception:
                pass
            try:
                del dfs_result
            except Exception:
                pass
            try:
                del tokens
            except Exception:
                pass
            gc.collect()
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            torch.cuda.reset_peak_memory_stats()

        except Exception as e:
            print(f"[Rank {rank}] ERROR on puzzle {key}: {type(e).__name__}: {e}")
            traceback.print_exc()
            if HAS_UNSLOTH:
                try:
                    model = FastLanguageModel.for_inference(model)
                except Exception:
                    pass
            for _n2 in ["puzzle_ds", "train_ds", "train_items", "puzzle_ds_multi", "eval_ds", "batches", "known_scores", "dfs_result", "tokens", "aug_dataset", "aug_queries", "aug_answers"]:
                try:
                    if _n2 == "puzzle_ds":
                        del puzzle_ds
                    elif _n2 == "train_ds":
                        del train_ds
                    elif _n2 == "train_items":
                        del train_items
                    elif _n2 == "puzzle_ds_multi":
                        del puzzle_ds_multi
                    elif _n2 == "eval_ds":
                        del eval_ds
                    elif _n2 == "batches":
                        del batches
                    elif _n2 == "known_scores":
                        del known_scores
                    elif _n2 == "dfs_result":
                        del dfs_result
                    elif _n2 == "tokens":
                        del tokens
                    elif _n2 == "aug_dataset":
                        del aug_dataset
                    elif _n2 == "aug_queries":
                        del aug_queries
                    elif _n2 == "aug_answers":
                        del aug_answers
                except Exception:
                    pass
            gc.collect()
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            torch.cuda.reset_peak_memory_stats()
            if isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.synchronize()

        spend_time = time.time() - start_time
        print(f"[Rank {rank}] finished {key} in {spend_time:.1f}s")
