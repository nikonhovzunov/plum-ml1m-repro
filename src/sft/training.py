from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .collator import SFTCollator


@dataclass(frozen=True)
class SFTTrainingConfig:
    model_name_or_path: str | Path
    output_dir: str | Path
    tokenizer_name_or_path: str | Path | None = None
    num_train_epochs: float = 1.0
    max_steps: int = -1
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.0
    warmup_steps: int = 0
    weight_decay: float = 0.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    eval_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 25
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    disable_tqdm: bool = True


def train_sft(
    train_dataset: Any,
    eval_dataset: Any | None,
    config: SFTTrainingConfig,
):
    try:
        from transformers import (
            GPT2LMHeadModel,
            GPT2TokenizerFast,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError("transformers is required for SFT training") from exc

    tokenizer_path = config.tokenizer_name_or_path or config.model_name_or_path
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(config.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    if config.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    args = TrainingArguments(
        **_training_args_kwargs(
            TrainingArguments,
            config,
            has_eval_dataset=eval_dataset is not None,
        )
    )
    collator = SFTCollator(tokenizer)

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
    }
    trainer_params = inspect.signature(Trainer).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    train_output = trainer.train()
    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))

    return {
        "trainer": trainer,
        "train_output": train_output,
        "tokenizer": tokenizer,
        "model": model,
    }


def _training_args_kwargs(
    training_args_cls,
    config: SFTTrainingConfig,
    has_eval_dataset: bool,
) -> dict[str, Any]:
    kwargs = {
        "output_dir": str(config.output_dir),
        "num_train_epochs": config.num_train_epochs,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "save_strategy": "steps",
        "save_steps": config.save_steps,
        "logging_steps": config.logging_steps,
        "fp16": config.fp16,
        "bf16": config.bf16,
        "report_to": "none",
        "save_total_limit": config.save_total_limit,
        "remove_unused_columns": False,
        "disable_tqdm": config.disable_tqdm,
    }
    if config.warmup_steps > 0:
        kwargs["warmup_steps"] = config.warmup_steps
    else:
        kwargs["warmup_ratio"] = config.warmup_ratio

    if has_eval_dataset:
        kwargs.update(
            {
                "eval_steps": config.eval_steps,
                "load_best_model_at_end": config.load_best_model_at_end,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            }
        )

    params = inspect.signature(training_args_cls).parameters
    if has_eval_dataset:
        if "eval_strategy" in params:
            kwargs["eval_strategy"] = "steps"
        else:
            kwargs["evaluation_strategy"] = "steps"

    return {key: value for key, value in kwargs.items() if key in params}
