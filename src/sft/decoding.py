from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from .mapping import SIDMapping
from .schema import SFTSchema

SID_RE = re.compile(r"^<sid_(\d+)_(\d+)>$")


def parse_sid_tokens(tokens: Iterable[str], n_levels: int = 5) -> tuple[int, ...] | None:
    values: list[int | None] = [None] * n_levels
    for token in tokens:
        match = SID_RE.match(token)
        if not match:
            continue
        level = int(match.group(1))
        code = int(match.group(2))
        if level >= n_levels or values[level] is not None:
            return None
        values[level] = code

    if any(value is None for value in values):
        return None
    return tuple(int(value) for value in values)


def build_allowed_level_token_ids(
    tokenizer: Any,
    schema: SFTSchema,
    sids: np.ndarray,
) -> dict[int, list[int]]:
    allowed: dict[int, set[int]] = {level: set() for level in range(schema.n_sid_levels)}
    for sid in np.asarray(sids):
        for level, token in enumerate(schema.cpt.sid_tokens(sid)):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id != tokenizer.unk_token_id:
                allowed[level].add(int(token_id))
    return {level: sorted(token_ids) for level, token_ids in allowed.items()}


def build_sid_trie(tokenizer: Any, schema: SFTSchema, sids: np.ndarray) -> dict[int, dict]:
    root: dict[int, dict] = {}
    eos = int(tokenizer.eos_token_id)
    for sid in np.asarray(sids):
        node = root
        for token_id in tokenizer.convert_tokens_to_ids(schema.cpt.sid_tokens(sid)):
            node = node.setdefault(int(token_id), {})
        node[eos] = {}
    return root


def level_prefix_allowed_tokens_fn(
    tokenizer: Any,
    allowed_by_level: dict[int, list[int]],
    prompt_length: int,
    n_sid_levels: int,
):
    def allowed_tokens(_batch_id: int, generated_ids: torch.Tensor) -> list[int]:
        generated_len = max(int(generated_ids.shape[-1]) - prompt_length, 0)
        if generated_len < n_sid_levels:
            return allowed_by_level[generated_len]
        return [int(tokenizer.eos_token_id)]

    return allowed_tokens


def trie_prefix_allowed_tokens_fn(
    tokenizer: Any,
    trie: dict[int, dict],
    prompt_length: int,
):
    def allowed_tokens(_batch_id: int, generated_ids: torch.Tensor) -> list[int]:
        node = trie
        for token_id in generated_ids[prompt_length:].tolist():
            token_id = int(token_id)
            if token_id not in node:
                return [int(tokenizer.eos_token_id)]
            node = node[token_id]
        return sorted(node.keys()) or [int(tokenizer.eos_token_id)]

    return allowed_tokens


def generate_sid_sequences(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    schema: SFTSchema,
    sids: np.ndarray | None = None,
    num_beams: int = 20,
    num_return_sequences: int = 20,
    constraint: str | None = None,
) -> list[dict[str, Any]]:
    prompt_length = int(input_ids.shape[-1])
    prefix_allowed_tokens_fn = None

    if constraint == "level":
        if sids is None:
            raise ValueError("sids are required for level-constrained decoding")
        allowed = build_allowed_level_token_ids(tokenizer, schema, sids)
        prefix_allowed_tokens_fn = level_prefix_allowed_tokens_fn(
            tokenizer, allowed, prompt_length, schema.n_sid_levels
        )
    elif constraint == "trie":
        if sids is None:
            raise ValueError("sids are required for trie-constrained decoding")
        trie = build_sid_trie(tokenizer, schema, sids)
        prefix_allowed_tokens_fn = trie_prefix_allowed_tokens_fn(
            tokenizer, trie, prompt_length
        )
    elif constraint is not None:
        raise ValueError(f"Unknown decoding constraint: {constraint}")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=schema.n_sid_levels + 1,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            do_sample=False,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

    decoded = []
    for sequence in outputs:
        new_ids = sequence[prompt_length:].tolist()
        tokens = tokenizer.convert_ids_to_tokens(new_ids)
        sid = parse_sid_tokens(tokens, n_levels=schema.n_sid_levels)
        decoded.append({"sid": sid, "tokens": tokens, "token_ids": new_ids})
    return decoded


def generate_recommendations(
    model: Any,
    tokenizer: Any,
    example: dict[str, Any],
    sid_mapping: SIDMapping,
    schema: SFTSchema,
    sids: np.ndarray,
    k: int = 20,
    num_beams: int = 50,
    constraint: str | None = "trie",
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    device = device or next(model.parameters()).device
    prompt_length = int(example["prompt_length"])
    prompt_ids = example["input_ids"][:prompt_length]

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    sid_outputs = generate_sid_sequences(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        schema=schema,
        sids=sids,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        constraint=constraint,
    )

    sid_candidates = [item["sid"] for item in sid_outputs]
    invalid_sid_count = sum(sid is None or not sid_mapping.has_sid(sid) for sid in sid_candidates)
    duplicate_count = len(sid_candidates) - len({sid for sid in sid_candidates if sid is not None})
    candidates = sid_mapping.sid_candidates_to_items(
        sid_candidates,
        k=k,
        seen_items=set(example.get("history_item_idx", [])),
        policy="expand",
    )

    return {
        "user_id": example.get("user_id"),
        "target_item_idx": int(example["target_item_idx"]),
        "target_sid": tuple(example["target_sid"]),
        "sid_outputs": sid_outputs,
        "candidates": candidates,
        "generated_sid_count": len(sid_candidates),
        "invalid_sid_count": invalid_sid_count,
        "duplicate_count": duplicate_count,
    }
