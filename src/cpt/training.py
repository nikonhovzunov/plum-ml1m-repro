from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CPTMixtureConfig:
    behavior_ratio: float = 0.5
    validation_size: float = 0.02
    seed: int = 42
    max_total_examples: int | None = None


@dataclass(frozen=True)
class GPT2CPTTrainingConfig:
    model_name_or_path: str = "gpt2"
    output_dir: str | Path = "data/processed/artifacts/CPT_user_behavior/cpt_gpt2"
    num_train_epochs: float = 1.0
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    max_steps: int = -1
    eval_steps: int = 200
    save_steps: int = 200
    logging_steps: int = 50
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = True
    overwrite_output_dir: bool = True
    save_total_limit: int = 2


def load_cpt_mixture(artifact_dir: str | Path, config: CPTMixtureConfig):
    try:
        from datasets import concatenate_datasets, load_from_disk
    except ImportError as exc:
        raise ImportError(
            "datasets is required to load CPT corpora. Install it in the notebook environment first."
        ) from exc

    artifact_dir = Path(artifact_dir)
    data_dir = artifact_dir / "data"

    behavior = load_from_disk(data_dir / "behavior").shuffle(seed=config.seed)
    meta_title = load_from_disk(data_dir / "meta_title")
    meta_genre = load_from_disk(data_dir / "meta_genre")
    meta_year = load_from_disk(data_dir / "meta_year")
    metadata = concatenate_datasets([meta_title, meta_genre, meta_year]).shuffle(
        seed=config.seed
    )

    if not 0.0 < config.behavior_ratio < 1.0:
        raise ValueError("behavior_ratio must be between 0 and 1")

    max_total = int(
        min(
            len(behavior) / config.behavior_ratio,
            len(metadata) / (1.0 - config.behavior_ratio),
        )
    )
    if config.max_total_examples is not None:
        max_total = min(max_total, int(config.max_total_examples))

    n_behavior = int(max_total * config.behavior_ratio)
    n_metadata = max_total - n_behavior

    mixed = concatenate_datasets(
        [
            behavior.select(range(n_behavior)),
            metadata.select(range(n_metadata)),
        ]
    ).shuffle(seed=config.seed)

    split = mixed.train_test_split(test_size=config.validation_size, seed=config.seed)
    return split["train"], split["test"], {
        "total": len(mixed),
        "behavior": n_behavior,
        "metadata": n_metadata,
        "train": len(split["train"]),
        "validation": len(split["test"]),
    }


def train_gpt2_cpt(
    artifact_dir: str | Path,
    training_config: GPT2CPTTrainingConfig,
    mixture_config: CPTMixtureConfig | None = None,
):
    try:
        from transformers import (
            DataCollatorForLanguageModeling,
            GPT2LMHeadModel,
            GPT2TokenizerFast,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError(
            "transformers is required for GPT-2 CPT training. "
            "Install it in the notebook environment first."
        ) from exc

    mixture_config = mixture_config or CPTMixtureConfig()
    artifact_dir = Path(artifact_dir)
    train_ds, val_ds, mixture_stats = load_cpt_mixture(artifact_dir, mixture_config)

    tokenizer = GPT2TokenizerFast.from_pretrained(artifact_dir / "tokenizer")
    model = GPT2LMHeadModel.from_pretrained(training_config.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    if training_config.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    args = TrainingArguments(**_training_args_kwargs(TrainingArguments, training_config))

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": collator,
    }
    trainer_params = inspect.signature(Trainer).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    train_output = trainer.train()
    trainer.save_model(str(training_config.output_dir))
    tokenizer.save_pretrained(str(training_config.output_dir))

    return {
        "trainer": trainer,
        "train_output": train_output,
        "mixture_stats": mixture_stats,
    }


def _training_args_kwargs(
    training_args_cls,
    config: GPT2CPTTrainingConfig,
) -> dict:
    kwargs = {
        "output_dir": str(config.output_dir),
        "overwrite_output_dir": config.overwrite_output_dir,
        "num_train_epochs": config.num_train_epochs,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "save_strategy": "steps",
        "eval_steps": config.eval_steps,
        "save_steps": config.save_steps,
        "logging_steps": config.logging_steps,
        "fp16": config.fp16,
        "bf16": config.bf16,
        "report_to": "none",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": config.save_total_limit,
    }

    params = inspect.signature(training_args_cls).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"

    return {key: value for key, value in kwargs.items() if key in params}
