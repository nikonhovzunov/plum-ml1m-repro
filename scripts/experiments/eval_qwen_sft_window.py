"""Legacy experimental evaluation launcher.

Not part of the canonical reproduction path. Kept for research history only;
use `plum-ml1m evaluate --config configs/evaluation_val.yaml` or
`configs/evaluation_test.yaml` for current protocol validation/planning.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(path: Path):
    try:
        return AutoTokenizer.from_pretrained(
            path, use_fast=True, fix_mistral_regex=True, local_files_only=True
        )
    except TypeError:
        return AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)


def load_causal_lm(path: Path, dtype: torch.dtype):
    kwargs = {"dtype": dtype, "local_files_only": True, "trust_remote_code": False}
    try:
        return AutoModelForCausalLM.from_pretrained(path, **kwargs)
    except TypeError:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        return AutoModelForCausalLM.from_pretrained(path, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--eval-history-len", type=int, required=True)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--beam-size", type=int, default=20)
    parser.add_argument("--num-return-sequences", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    root = args.root.resolve()
    artifact_dir = args.artifact_dir.resolve()
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    best_adapter_dir = artifact_dir / "best_adapter"
    if not best_adapter_dir.exists():
        summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
        best_adapter_dir = Path(summary["best_adapter_dir"])

    base_cpt_dir = Path(config["base_cpt_dir"])
    sid_array_path = Path(config["sid_array_path"])
    train_path = root / "data/processed/splits/train.parquet"
    val_path = root / "data/processed/splits/val.parquet"
    users_path = root / "data/raw/ml-1m/users.dat"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.is_bf16_supported()
        else (torch.float16 if device == "cuda" else torch.float32)
    )
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.empty_cache()

    tokenizer = load_tokenizer(best_adapter_dir)
    sids = np.load(sid_array_path)

    train = pd.read_parquet(train_path).sort_values(
        ["user_idx", "timestamp", "pos", "item_idx"], kind="mergesort"
    )
    val = pd.read_parquet(val_path).sort_values(
        ["user_idx", "timestamp", "pos", "item_idx"], kind="mergesort"
    )
    users = pd.read_csv(
        users_path,
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"],
        encoding="latin-1",
    )
    users_by_id = users.set_index("user_id", drop=False)

    bos, eos = "<bos>", "<eos>"
    user_open, user_close = "<user>", "</user>"
    hist, event_open, event_close = "<hist>", "<e>", "</e>"
    task_rec, next_token = "<task_rec>", "<next>"

    def base_sid_tokens(sid):
        return [f"<sid_{level}_{int(code)}>" for level, code in enumerate(sid)]

    sid_keys = [tuple(int(x) for x in sid) for sid in sids]
    sid_to_items: dict[tuple[int, ...], list[int]] = {}
    for item_idx, key in enumerate(sid_keys):
        sid_to_items.setdefault(key, []).append(int(item_idx))

    item_dup_suffix: dict[int, str] = {}
    for _key, items in sid_to_items.items():
        if len(items) > 1:
            for dup_i, item_idx in enumerate(sorted(items)):
                item_dup_suffix[int(item_idx)] = f"<dup_{dup_i}>"

    def item_sid_tokens(item_idx: int):
        tokens = base_sid_tokens(sids[int(item_idx)])
        if int(item_idx) in item_dup_suffix:
            tokens = tokens + [item_dup_suffix[int(item_idx)]]
        return tokens

    def token_ids_for_item(item_idx: int):
        return tuple(int(x) for x in tokenizer.convert_tokens_to_ids(item_sid_tokens(item_idx)))

    item_to_token_ids = {int(i): token_ids_for_item(i) for i in range(len(sids))}
    token_ids_to_item = {v: k for k, v in item_to_token_ids.items()}
    if len(token_ids_to_item) != len(item_to_token_ids):
        raise RuntimeError("item token sequences are not one-to-one")

    trie: dict[int, dict] = {}
    eos_id = int(tokenizer.eos_token_id)
    pad_id = int(tokenizer.pad_token_id)
    for ids in item_to_token_ids.values():
        node = trie
        for token_id in ids:
            node = node.setdefault(int(token_id), {})
        node[eos_id] = {}

    def rating_token(rating):
        return f"<rat_{int(rating)}>"

    def user_tokens(row):
        return [f"<gen_{row.gender}>", f"<age_{int(row.age)}>", f"<occ_{int(row.occupation)}>"]

    def event_tokens(event):
        tokens = [event_open]
        tokens.extend(item_sid_tokens(int(event["item_idx"])))
        tokens.append(rating_token(event["rating"]))
        tokens.append(event_close)
        return tokens

    def prompt_prefix_tokens(user_id):
        tokens = [bos, task_rec]
        if int(user_id) in users_by_id.index:
            tokens.append(user_open)
            tokens.extend(user_tokens(users_by_id.loc[int(user_id)]))
            tokens.append(user_close)
        tokens.append(hist)
        return tokens

    max_seq_length = int(config["max_seq_length"])

    def fit_prompt(prefix, history_events, target_tokens):
        event_blocks = [event_tokens(event) for event in history_events]
        while event_blocks:
            prompt_tokens = prefix + [tok for block in event_blocks for tok in block] + [next_token]
            if len(prompt_tokens) + len(target_tokens) <= max_seq_length:
                return prompt_tokens
            event_blocks = event_blocks[1:]
        prompt_tokens = prefix + [next_token]
        if len(prompt_tokens) + len(target_tokens) > max_seq_length:
            raise ValueError("Prompt prefix plus target exceeds max_seq_length")
        return prompt_tokens

    def encode_example(user_id, history_events, target_event):
        target_item = int(target_event["item_idx"])
        target_tokens = item_sid_tokens(target_item) + [eos]
        prompt_tokens = fit_prompt(prompt_prefix_tokens(user_id), history_events, target_tokens)
        all_tokens = prompt_tokens + target_tokens
        input_ids = tokenizer.convert_tokens_to_ids(all_tokens)
        if any(token_id == tokenizer.unk_token_id for token_id in input_ids):
            bad = [
                tok
                for tok, token_id in zip(all_tokens, input_ids, strict=False)
                if token_id == tokenizer.unk_token_id
            ]
            raise ValueError(f"Unknown tokens: {bad[:10]}")
        return {
            "input_ids": input_ids,
            "prompt_length": len(prompt_tokens),
            "user_id": int(user_id),
            "target_item_idx": target_item,
            "history_item_idx": [int(event["item_idx"]) for event in history_events],
        }

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

    examples = []
    for user_id in sorted(val_target_by_user):
        full_history = train_events_by_user.get(int(user_id), [])
        history_events = full_history[-args.eval_history_len :]
        if len(history_events) < args.eval_history_len:
            continue
        examples.append(encode_example(user_id, history_events, val_target_by_user[user_id]))

    print(
        json.dumps(
            {
                "artifact_dir": str(artifact_dir),
                "best_adapter_dir": str(best_adapter_dir),
                "eval_history_len": args.eval_history_len,
                "examples": len(examples),
                "device": device,
                "dtype": str(dtype),
            },
            indent=2,
        )
    )

    base = load_causal_lm(base_cpt_dir, dtype=dtype)
    base.resize_token_embeddings(len(tokenizer))
    base.config.bos_token_id = tokenizer.bos_token_id
    base.config.eos_token_id = tokenizer.eos_token_id
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, best_adapter_dir).to(device)
    model.eval()
    model.config.use_cache = True

    def trie_allowed_tokens(prompt_length):
        def allowed(_batch_id, input_ids):
            node = trie
            for token_id in input_ids[prompt_length:].tolist():
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

    def generate_candidates(example):
        prompt_ids = example["input_ids"][: example["prompt_length"]]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        seen = set(example["history_item_idx"])

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=8,
                num_beams=args.beam_size,
                num_return_sequences=min(args.num_return_sequences, args.beam_size),
                do_sample=False,
                early_stopping=True,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                prefix_allowed_tokens_fn=trie_allowed_tokens(len(prompt_ids)),
                use_cache=True,
            )

        candidates = []
        invalid = 0
        seen_generated = 0
        used = set()
        for seq in outputs:
            item = decode_generated_item(seq, len(prompt_ids))
            if item is None:
                invalid += 1
                continue
            item = int(item)
            if item in seen:
                seen_generated += 1
                continue
            if item in used:
                continue
            candidates.append(item)
            used.add(item)
            if len(candidates) >= args.top_k:
                break
        return candidates, invalid, seen_generated, int(len(outputs))

    def recall_at_k(candidates, target, k):
        return float(int(int(target) in candidates[:k]))

    def ndcg_at_k(candidates, target, k):
        target = int(target)
        for rank, item in enumerate(candidates[:k], start=1):
            if int(item) == target:
                return 1.0 / math.log2(rank + 1)
        return 0.0

    def mrr_at_k(candidates, target, k):
        target = int(target)
        for rank, item in enumerate(candidates[:k], start=1):
            if int(item) == target:
                return 1.0 / rank
        return 0.0

    metrics = {"split": f"val_full_best_w{args.eval_history_len}", "n": len(examples)}
    recommended = set()
    invalid = 0
    generated = 0
    seen_generated = 0
    candidate_counts = []

    for ex in tqdm(examples, desc=f"eval val_w{args.eval_history_len}"):
        candidates, inv, seen_gen, gen_count = generate_candidates(ex)
        target = int(ex["target_item_idx"])
        recommended.update(candidates)
        candidate_counts.append(len(candidates))
        invalid += inv
        generated += gen_count
        seen_generated += seen_gen
        for k in [1, 5, 10]:
            metrics[f"recall@{k}"] = metrics.get(f"recall@{k}", 0.0) + recall_at_k(
                candidates, target, k
            )
            metrics[f"ndcg@{k}"] = metrics.get(f"ndcg@{k}", 0.0) + ndcg_at_k(candidates, target, k)
            metrics[f"mrr@{k}"] = metrics.get(f"mrr@{k}", 0.0) + mrr_at_k(candidates, target, k)

    n = max(len(examples), 1)
    for key in list(metrics):
        if key.startswith(("recall@", "ndcg@", "mrr@")):
            metrics[key] = metrics[key] / n
    metrics["coverage@10"] = int(len(recommended))
    metrics["avg_candidates"] = float(np.mean(candidate_counts)) if candidate_counts else 0.0
    metrics["invalid_sid_rate"] = float(invalid / max(generated, 1))
    metrics["seen_generated_rate"] = float(seen_generated / max(generated, 1))

    output_name = args.output_name or f"full_val_metrics_window{args.eval_history_len}.json"
    output_path = artifact_dir / output_name
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, indent=2))
    print("saved:", output_path)


if __name__ == "__main__":
    main()
