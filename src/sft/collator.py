from __future__ import annotations

from typing import Any

import torch


class SFTCollator:
    def __init__(self, tokenizer: Any):
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have pad_token_id for SFT batching")
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = int(self.tokenizer.pad_token_id)

        input_ids = []
        attention_mask = []
        labels = []

        for feature in features:
            length = len(feature["input_ids"])
            pad = max_len - length
            input_ids.append(feature["input_ids"] + [pad_id] * pad)
            attention_mask.append([1] * length + [0] * pad)
            labels.append(feature["labels"] + [-100] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
