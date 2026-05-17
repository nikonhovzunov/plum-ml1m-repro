"""Experimental Qwen3 SFT beam sweep / final split evaluator.

Kept as an experiment launcher for the Qwen3 validation/test series. Canonical
protocol components live under `plum_ml1m`; use configs for split-specific
validation and document any new metrics produced by this script.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def find_root(start: Path) -> Path:
    root = start.resolve()
    while not (root / "src").exists() and root.parent != root:
        root = root.parent
    return root


ROOT = find_root(Path.cwd())
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

evaluate_rankings = importlib.import_module("plum_ml1m.metrics").evaluate_rankings

RUN_NAME = "sft_qwen3_4b_qlora32_sid_v2_next_watch_w16_allseen_pat3_v1"
BASE_CPT_DIR = (
    ROOT / "data/processed/artifacts/cpt_qwen3_4b_base_sid_v2_plum_curriculum_qlora32_v1/final_merged"
)
OUTPUT_DIR = ROOT / "data/processed/artifacts" / RUN_NAME
BEST_ADAPTER_DIR = OUTPUT_DIR / "best_adapter"
SWEEP_DIR = OUTPUT_DIR / "beam_sweep"

SID_ARRAY_PATH = ROOT / "runs/qwen4b_rqvae_sid_v2_plum/SIDs_best.npy"
TRAIN_PATH = ROOT / "data/processed/splits/train.parquet"
VAL_PATH = ROOT / "data/processed/splits/val.parquet"
TEST_PATH = ROOT / "data/processed/splits/test.parquet"
USERS_PATH = ROOT / "data/raw/ml-1m/users.dat"

SEED = 42
MIN_HISTORY_LEN = 16
EVAL_HISTORY_LEN = 16
MAX_SEQ_LENGTH = 192
MAX_TARGET_TOKENS = 8
INCLUDE_RATINGS = True
INCLUDE_USER_FEATURES = True
FILTER_SEEN = True
STILL_ACTIVE = 259

BOS = "<bos>"
EOS = "<eos>"
PAD = "<pad>"
USER_OPEN = "<user>"
USER_CLOSE = "</user>"
HIST = "<hist>"
EVENT_OPEN = "<e>"
EVENT_CLOSE = "</e>"
TASK_REC = "<task_rec>"
NEXT = "<next>"


def load_tokenizer(path: Path):
    try:
        return AutoTokenizer.from_pretrained(path, local_files_only=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(path)


def load_causal_lm(path: Path, dtype: torch.dtype, load_in_4bit: bool = False):
    kwargs = {"local_files_only": True, "trust_remote_code": False}
    if load_in_4bit:
        kwargs.update(
            {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                ),
                "device_map": {"": 0} if torch.cuda.is_available() else None,
                "low_cpu_mem_usage": True,
            }
        )
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
    else:
        kwargs["dtype"] = dtype
    try:
        return AutoModelForCausalLM.from_pretrained(path, **kwargs)
    except TypeError:
        if "dtype" in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        return AutoModelForCausalLM.from_pretrained(path, **kwargs)


def base_sid_tokens(sid):
    return [f"<sid_{level}_{int(code)}>" for level, code in enumerate(sid)]


def rating_token(rating):
    return f"<rat_{int(rating)}>"


def user_tokens(row):
    return [f"<gen_{row.gender}>", f"<age_{int(row.age)}>", f"<occ_{int(row.occupation)}>"]


def is_windows_pid_alive(pid: int) -> bool:
    kernel32 = ctypes.windll.kernel32
    process_query_limited_information = 0x1000
    handle = kernel32.OpenProcess(process_query_limited_information, False, int(pid))
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return int(exit_code.value) == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def wait_for_pids(pids: list[int], poll_seconds: int, output_dir: Path) -> None:
    if not pids:
        return
    status_path = output_dir / "wait_status.log"
    while True:
        alive = [pid for pid in pids if is_windows_pid_alive(pid)]
        with status_path.open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} alive={alive}\n")
        if not alive:
            return
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-users", type=int, default=256)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--configs", default="20:20,40:40,100:100")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=SWEEP_DIR)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--base-cpt-dir", type=Path, default=BASE_CPT_DIR)
    parser.add_argument("--adapter-dir", type=Path, default=BEST_ADAPTER_DIR)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--seen-filter-scope",
        choices=["prompt", "all"],
        default="all",
        help=(
            "prompt filters only the visible prompt history; all filters every item "
            "seen before the validation/test target."
        ),
    )
    parser.add_argument("--wait-pids", default="")
    parser.add_argument("--wait-poll-seconds", type=int, default=60)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_cpt_dir = args.base_cpt_dir.resolve()
    adapter_dir = args.adapter_dir.resolve()
    wait_pids = [int(x) for x in args.wait_pids.split(",") if x.strip()]
    wait_for_pids(wait_pids, args.wait_poll_seconds, args.output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else (torch.float16 if device == "cuda" else torch.float32)

    train = pd.read_parquet(TRAIN_PATH).sort_values(
        ["user_idx", "timestamp", "pos", "item_idx"], kind="mergesort"
    )
    val = pd.read_parquet(VAL_PATH).sort_values(
        ["user_idx", "timestamp", "pos", "item_idx"], kind="mergesort"
    )
    test = pd.read_parquet(TEST_PATH).sort_values(
        ["user_idx", "timestamp", "pos", "item_idx"], kind="mergesort"
    )
    users = pd.read_csv(
        USERS_PATH,
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"],
        encoding="latin-1",
    )
    sids = np.load(SID_ARRAY_PATH)

    tokenizer = load_tokenizer(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sid_keys = [tuple(int(x) for x in sid) for sid in sids]
    sid_to_items: dict[tuple[int, ...], list[int]] = {}
    for item_idx, key in enumerate(sid_keys):
        sid_to_items.setdefault(key, []).append(int(item_idx))

    item_dup_suffix: dict[int, str] = {}
    for _key, items in sid_to_items.items():
        if len(items) > 1:
            for dup_i, item_idx in enumerate(sorted(items)):
                item_dup_suffix[int(item_idx)] = f"<dup_{dup_i}>"

    def item_sid_tokens(item_idx: int) -> list[str]:
        tokens = base_sid_tokens(sids[int(item_idx)])
        if int(item_idx) in item_dup_suffix:
            tokens = tokens + [item_dup_suffix[int(item_idx)]]
        return tokens

    def token_ids_for_item(item_idx: int) -> tuple[int, ...]:
        return tuple(int(x) for x in tokenizer.convert_tokens_to_ids(item_sid_tokens(item_idx)))

    item_to_token_ids = {int(i): token_ids_for_item(i) for i in range(len(sids))}
    token_ids_to_item = {v: k for k, v in item_to_token_ids.items()}

    trie: dict[int, dict] = {}
    for ids in item_to_token_ids.values():
        node = trie
        for token_id in ids:
            node = node.setdefault(int(token_id), {})
        node[int(tokenizer.eos_token_id)] = {}

    eos_id = int(tokenizer.eos_token_id)
    pad_id = int(tokenizer.pad_token_id)

    users_by_id = users.set_index("user_id", drop=False)
    train_events_by_user = {
        int(user_id): group.sort_values(["timestamp", "pos", "item_idx"], kind="mergesort").to_dict(
            "records"
        )
        for user_id, group in train.groupby("user_id", sort=False)
    }
    val_target_by_user = {
        int(row.user_id): row._asdict()
        for row in val.sort_values(
            ["user_idx", "timestamp", "pos", "item_idx"], kind="mergesort"
        ).itertuples(index=False)
    }
    test_target_by_user = {
        int(row.user_id): row._asdict()
        for row in test.sort_values(
            ["user_idx", "timestamp", "pos", "item_idx"], kind="mergesort"
        ).itertuples(index=False)
    }
    train_plus_val_events_by_user = {
        int(user_id): group.sort_values(["timestamp", "pos", "item_idx"], kind="mergesort").to_dict(
            "records"
        )
        for user_id, group in pd.concat([train, val], ignore_index=True).groupby(
            "user_id", sort=False
        )
    }

    def event_tokens(event) -> list[str]:
        tokens = [EVENT_OPEN]
        tokens.extend(item_sid_tokens(int(event["item_idx"])))
        if INCLUDE_RATINGS:
            tokens.append(rating_token(event["rating"]))
        tokens.append(EVENT_CLOSE)
        return tokens

    def prompt_prefix_tokens(user_id: int) -> list[str]:
        tokens = [BOS, TASK_REC]
        if INCLUDE_USER_FEATURES and int(user_id) in users_by_id.index:
            tokens.append(USER_OPEN)
            tokens.extend(user_tokens(users_by_id.loc[int(user_id)]))
            tokens.append(USER_CLOSE)
        tokens.append(HIST)
        return tokens

    def fit_prompt(prefix, history_events, target_tokens):
        event_blocks = [event_tokens(event) for event in history_events]
        while event_blocks:
            prompt_tokens = prefix + [tok for block in event_blocks for tok in block] + [NEXT]
            if len(prompt_tokens) + len(target_tokens) <= MAX_SEQ_LENGTH:
                return prompt_tokens
            event_blocks = event_blocks[1:]
        return prefix + [NEXT]

    def encode_example(user_id: int, history_events, target_event, split: str, seen_events=None):
        seen_events = history_events if seen_events is None else seen_events
        target_item = int(target_event["item_idx"])
        target_tokens = item_sid_tokens(target_item) + [EOS]
        prompt_tokens = fit_prompt(prompt_prefix_tokens(user_id), history_events, target_tokens)
        all_tokens = prompt_tokens + target_tokens
        input_ids = tokenizer.convert_tokens_to_ids(all_tokens)
        return {
            "input_ids": input_ids,
            "prompt_length": len(prompt_tokens),
            "user_id": int(user_id),
            "split": split,
            "target_item_idx": target_item,
            "history_item_idx": [int(event["item_idx"]) for event in history_events],
            "all_seen_item_idx": [int(event["item_idx"]) for event in seen_events],
        }

    def build_eval_examples(split: str, max_users: int | None):
        if split == "val":
            context_by_user = train_events_by_user
            target_by_user = val_target_by_user
        else:
            context_by_user = train_plus_val_events_by_user
            target_by_user = test_target_by_user
        user_ids = sorted(target_by_user.keys())
        if max_users is not None:
            rng = np.random.default_rng(SEED)
            user_ids = sorted(
                rng.choice(user_ids, size=min(max_users, len(user_ids)), replace=False).tolist()
            )
        examples = []
        for user_id in user_ids:
            full_history = context_by_user.get(int(user_id), [])
            history = full_history[-EVAL_HISTORY_LEN:]
            if len(history) < MIN_HISTORY_LEN:
                continue
            examples.append(
                encode_example(
                    int(user_id),
                    history,
                    target_by_user[int(user_id)],
                    split,
                    seen_events=full_history,
                )
            )
        return examples

    max_users = None if args.max_users is not None and args.max_users <= 0 else args.max_users
    examples = build_eval_examples(args.split, max_users)
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")

    model = load_causal_lm(base_cpt_dir, dtype, load_in_4bit=args.load_in_4bit)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True
    model = PeftModel.from_pretrained(model, adapter_dir)
    if not args.load_in_4bit:
        model = model.to(device)
    model.eval()

    def trie_allowed_tokens(prompt_length):
        def allowed(_batch_id, input_ids):
            node = trie
            generated = input_ids[prompt_length:].tolist()
            for token_id in generated:
                token_id = int(token_id)
                if token_id not in node:
                    return [eos_id]
                node = node[token_id]
            return sorted(node.keys()) if node else [eos_id]

        return allowed

    def batched_trie_allowed_tokens(start_lengths: list[int]):
        def allowed(batch_id, input_ids):
            node = trie
            start = int(start_lengths[int(batch_id)])
            generated = input_ids[start:].tolist()
            for token_id in generated:
                token_id = int(token_id)
                if token_id not in node:
                    return [eos_id]
                node = node[token_id]
            return sorted(node.keys()) if node else [eos_id]

        return allowed

    def decode_generated_item(sequence_ids, prompt_length):
        new_ids = []
        for token_id in sequence_ids[prompt_length:].tolist():
            token_id = int(token_id)
            if token_id in {eos_id, pad_id}:
                break
            new_ids.append(token_id)
        return token_ids_to_item.get(tuple(new_ids))

    def generate_candidates(example, beam_size: int, num_return_sequences: int):
        prompt_ids = example["input_ids"][: example["prompt_length"]]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        seen_field = "all_seen_item_idx" if args.seen_filter_scope == "all" else "history_item_idx"
        seen = set(example.get(seen_field, [])) if FILTER_SEEN else set()
        num_return_sequences = min(int(num_return_sequences), int(beam_size))

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_TARGET_TOKENS,
                num_beams=int(beam_size),
                num_return_sequences=int(num_return_sequences),
                do_sample=False,
                early_stopping=True,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                prefix_allowed_tokens_fn=trie_allowed_tokens(len(prompt_ids)),
                use_cache=True,
            )

        candidates = []
        raw_items = []
        invalid = 0
        seen_generated = 0
        used = set()
        for seq in outputs:
            item = decode_generated_item(seq, len(prompt_ids))
            if item is None:
                invalid += 1
                continue
            raw_items.append(int(item))
            if item in seen:
                seen_generated += 1
                continue
            if item in used:
                continue
            used.add(int(item))
            candidates.append(int(item))

        return {
            "candidates": candidates,
            "raw_items": raw_items,
            "invalid_sid_count": invalid,
            "seen_generated_count": seen_generated,
            "generated_count": int(len(outputs)),
        }

    def generate_batch(batch, beam_size: int, num_return_sequences: int):
        prompt_ids_list = [ex["input_ids"][: ex["prompt_length"]] for ex in batch]
        max_prompt_len = max(len(ids) for ids in prompt_ids_list)
        batch_input_ids = torch.full(
            (len(batch), max_prompt_len),
            fill_value=pad_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(batch_input_ids)
        for row_idx, prompt_ids in enumerate(prompt_ids_list):
            ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
            batch_input_ids[row_idx, -len(prompt_ids) :] = ids
            attention_mask[row_idx, -len(prompt_ids) :] = 1

        num_return_sequences = min(int(num_return_sequences), int(beam_size))
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_TARGET_TOKENS,
                num_beams=int(beam_size),
                num_return_sequences=int(num_return_sequences),
                do_sample=False,
                early_stopping=True,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                prefix_allowed_tokens_fn=batched_trie_allowed_tokens([max_prompt_len] * len(batch)),
                use_cache=True,
            )

        seen_field = "all_seen_item_idx" if args.seen_filter_scope == "all" else "history_item_idx"
        results = []
        for row_idx, example in enumerate(batch):
            candidates = []
            raw_items = []
            invalid = 0
            seen_generated = 0
            used = set()
            seen = set(example.get(seen_field, [])) if FILTER_SEEN else set()
            start = row_idx * num_return_sequences
            end = start + num_return_sequences
            for seq in outputs[start:end]:
                item = decode_generated_item(seq, max_prompt_len)
                if item is None:
                    invalid += 1
                    continue
                raw_items.append(int(item))
                if item in seen:
                    seen_generated += 1
                    continue
                if item in used:
                    continue
                used.add(int(item))
                candidates.append(int(item))
            results.append(
                {
                    "candidates": candidates,
                    "raw_items": raw_items,
                    "invalid_sid_count": invalid,
                    "seen_generated_count": seen_generated,
                    "generated_count": int(num_return_sequences),
                }
            )
        return results

    def evaluate_config(beam_size: int, num_return_sequences: int):
        started = time.time()
        records = []
        batches = [
            examples[i : i + args.eval_batch_size]
            for i in range(0, len(examples), args.eval_batch_size)
        ]
        for batch in tqdm(
            batches,
            desc=f"{args.split} beam={beam_size} return={num_return_sequences}",
            leave=False,
        ):
            if args.eval_batch_size == 1:
                batch_generations = [
                    generate_candidates(
                        batch[0], beam_size=beam_size, num_return_sequences=num_return_sequences
                    )
                ]
            else:
                batch_generations = generate_batch(
                    batch, beam_size=beam_size, num_return_sequences=num_return_sequences
                )
            for ex, gen in zip(batch, batch_generations, strict=True):
                records.append({"target_item_idx": int(ex["target_item_idx"]), **gen})

        metrics = {
            "split": args.split,
            "n": len(records),
            "beam_size": int(beam_size),
            "num_return_sequences": int(num_return_sequences),
            "seconds": time.time() - started,
        }
        metrics.update(
            evaluate_rankings(
                [
                    {
                        "target_item_idx": int(rec["target_item_idx"]),
                        "candidates": rec["candidates"],
                    }
                    for rec in records
                ],
                k_values=(1, 5, 10, 20, 50, 100),
            )
        )
        sums = {
            "generated": 0,
            "invalid": 0,
            "seen": 0,
            "raw_valid": 0,
            "raw_duplicate_valid": 0,
            "unique_filtered": 0,
        }
        for rec in records:
            candidates = rec["candidates"]
            raw_items = rec["raw_items"]
            raw_unique = set(raw_items)
            sums["generated"] += int(rec["generated_count"])
            sums["invalid"] += int(rec["invalid_sid_count"])
            sums["seen"] += int(rec["seen_generated_count"])
            sums["raw_valid"] += len(raw_items)
            sums["raw_duplicate_valid"] += max(0, len(raw_items) - len(raw_unique))
            sums["unique_filtered"] += len(candidates)

        n = max(len(records), 1)

        generated = max(sums["generated"], 1)
        raw_valid = max(sums["raw_valid"], 1)
        metrics.update(
            {
                "raw_valid_rate": (sums["raw_valid"] / generated),
                "invalid_sid_rate": (sums["invalid"] / generated),
                "seen_generated_rate": (sums["seen"] / generated),
                "raw_duplicate_valid_rate": (sums["raw_duplicate_valid"] / raw_valid),
                "avg_raw_valid": sums["raw_valid"] / n,
                "avg_unique_filtered_candidates": sums["unique_filtered"] / n,
            }
        )
        return metrics

    run_config = {
        "run_name": args.run_name,
        "base_cpt_dir": str(base_cpt_dir),
        "adapter_dir": str(adapter_dir),
        "split": args.split,
        "max_users": max_users,
        "examples": len(examples),
        "configs": args.configs,
        "eval_batch_size": args.eval_batch_size,
        "seen_filter_scope": args.seen_filter_scope,
        "filter_seen": FILTER_SEEN,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (args.output_dir / "beam_sweep_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    results = []
    results_jsonl = args.output_dir / "beam_sweep_results.jsonl"
    for spec in args.configs.split(","):
        beam_s, return_s = spec.split(":")
        beam_size = int(beam_s)
        num_return_sequences = int(return_s)
        try:
            result = evaluate_config(beam_size, num_return_sequences)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            result = {
                "split": args.split,
                "n": len(examples),
                "beam_size": beam_size,
                "num_return_sequences": num_return_sequences,
                "error": f"CUDA OOM: {exc}",
            }
        results.append(result)
        with results_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        pd.DataFrame(results).to_csv(args.output_dir / "beam_sweep_results.csv", index=False)
        (args.output_dir / "beam_sweep_results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
