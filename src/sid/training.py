"""Training utilities for advanced PLUM-style RQ-VAE experiments."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .features import FeatureBundle
from .rqvae import AdvancedRQVAE, AdvancedRQVAELoss, RQVAEConfig, count_parameters


@dataclass(frozen=True)
class AdvancedRQVAETrainingConfig:
    seed: int = 42
    device: str = "cuda"
    epochs: int = 80
    steps_per_epoch: int = 64
    item_batch_size: int = 512
    pair_batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-4
    beta: float = 0.25
    contrastive_weight: float = 0.05
    contrastive_temperature: float = 0.07
    max_grad_norm: float = 1.0
    progressive_masking: bool = True
    progressive_warmup_epochs: int = 24
    progressive_schedule: tuple[tuple[int, int], ...] | None = None
    progressive_sampling: str = "deterministic"
    eval_every: int = 5
    eval_pair_sample: int = 10000
    sid_eval_batch_size: int = 1024
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    show_progress: bool = True
    num_workers: int = 0


class ItemFeatureDataset(Dataset):
    def __init__(self, bundle: FeatureBundle):
        self.meta = torch.from_numpy(np.array(bundle.meta, copy=True))
        self.description = torch.from_numpy(np.array(bundle.description, copy=True))
        self.description_mask = torch.from_numpy(np.array(bundle.description_mask, copy=True))

    def __len__(self) -> int:
        return int(self.meta.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "inputs": {
                "meta": self.meta[idx],
                "description": self.description[idx],
            },
            "description_mask": self.description_mask[idx],
        }


class BehaviorPairDataset(Dataset):
    def __init__(self, bundle: FeatureBundle, pairs: pd.DataFrame):
        required = {"item_idx", "item_pos", "score"}
        missing = required - set(pairs.columns)
        if missing:
            raise ValueError(f"pairs is missing columns: {sorted(missing)}")

        item_idx = bundle.item_idx
        if not np.array_equal(item_idx, np.arange(len(item_idx))):
            raise ValueError("Feature bundle must be sorted and dense by item_idx.")

        self.meta = torch.from_numpy(np.array(bundle.meta, copy=True))
        self.description = torch.from_numpy(np.array(bundle.description, copy=True))
        self.description_mask = torch.from_numpy(np.array(bundle.description_mask, copy=True))
        self.anchor = torch.from_numpy(np.array(pairs["item_idx"].to_numpy(np.int64), copy=True))
        self.positive = torch.from_numpy(np.array(pairs["item_pos"].to_numpy(np.int64), copy=True))
        score = pairs["score"].to_numpy(np.float32)
        score = score / max(float(np.mean(score)), 1e-8)
        self.weight = torch.from_numpy(np.array(score.astype(np.float32), copy=True))

    def __len__(self) -> int:
        return int(self.anchor.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        anchor = int(self.anchor[idx])
        positive = int(self.positive[idx])
        return {
            "anchor_inputs": {
                "meta": self.meta[anchor],
                "description": self.description[anchor],
            },
            "anchor_description_mask": self.description_mask[anchor],
            "positive_inputs": {
                "meta": self.meta[positive],
                "description": self.description[positive],
            },
            "positive_description_mask": self.description_mask[positive],
            "weight": self.weight[idx],
        }


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(val, device) for key, val in value.items()}
    return value


def _max_active_levels_for_epoch(
    epoch: int,
    n_levels: int,
    config: AdvancedRQVAETrainingConfig,
) -> int:
    if not config.progressive_masking:
        return n_levels

    if config.progressive_schedule:
        active_levels = 1
        for start_epoch, levels in sorted(config.progressive_schedule):
            if start_epoch < 1:
                raise ValueError("progressive_schedule epochs must be >= 1.")
            if not 1 <= levels <= n_levels:
                raise ValueError(
                    f"progressive_schedule levels must be in [1, {n_levels}]."
                )
            if epoch >= start_epoch:
                active_levels = levels
        return int(active_levels)

    warmup = max(config.progressive_warmup_epochs, 1)
    epochs_per_level = max(1, math.ceil(warmup / n_levels))
    return min(n_levels, 1 + (epoch - 1) // epochs_per_level)


def _sample_active_levels(
    max_active_levels: int,
    n_levels: int,
    config: AdvancedRQVAETrainingConfig,
) -> int:
    if not config.progressive_masking:
        return n_levels

    mode = config.progressive_sampling.lower().strip()
    if mode in {"deterministic", "fixed", "max"}:
        return int(max_active_levels)
    if mode == "uniform_prefix":
        return random.randint(1, int(max_active_levels))
    raise ValueError(
        "progressive_sampling must be 'deterministic' or 'uniform_prefix'."
    )


def _active_levels_for_epoch(
    epoch: int,
    n_levels: int,
    config: AdvancedRQVAETrainingConfig,
) -> int:
    max_active_levels = _max_active_levels_for_epoch(epoch, n_levels, config)
    return _sample_active_levels(max_active_levels, n_levels, config)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def encode_all_items(
    model: AdvancedRQVAE,
    bundle: FeatureBundle,
    device: torch.device,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    sids: list[np.ndarray] = []
    contrastive: list[np.ndarray] = []
    z_q: list[np.ndarray] = []
    dataset = ItemFeatureDataset(bundle)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            output = model(
                batch["inputs"],
                description_mask=batch["description_mask"],
                active_levels=model.config.n_levels,
            )
            sids.append(output.sids.detach().cpu().numpy())
            contrastive.append(output.contrastive.detach().cpu().numpy())
            z_q.append(output.z_q_st.detach().cpu().numpy())
    return np.vstack(sids), np.vstack(contrastive), np.vstack(z_q)


def sid_metrics(sids: np.ndarray, codebook_sizes: tuple[int, ...]) -> dict[str, Any]:
    _, counts = np.unique(sids, axis=0, return_counts=True)
    unique = int(counts.shape[0])
    n_items = int(sids.shape[0])
    entropy = 0.0
    probs = counts / counts.sum()
    entropy = float(-(probs * np.log(probs)).sum() / np.log(max(n_items, 2)))
    per_level_unique = []
    per_level_unique_fraction_codebook = []
    per_level_unique_fraction_items = []
    per_level_perplexity = []
    per_level_entropy = []
    per_depth_unique_sids = []
    per_depth_sid_uniqueness = []
    for level, size in enumerate(codebook_sizes):
        ids = sids[:, level]
        level_counts = np.bincount(ids, minlength=size)
        p = level_counts[level_counts > 0] / level_counts.sum()
        h = float(-(p * np.log(p)).sum())
        level_unique = int((level_counts > 0).sum())
        per_level_unique.append(level_unique)
        per_level_unique_fraction_codebook.append(float(level_unique / max(size, 1)))
        per_level_unique_fraction_items.append(float(level_unique / max(n_items, 1)))
        per_level_perplexity.append(float(np.exp(h)))
        per_level_entropy.append(float(h / np.log(max(size, 2))))
        depth_unique = int(np.unique(sids[:, : level + 1], axis=0).shape[0])
        per_depth_unique_sids.append(depth_unique)
        per_depth_sid_uniqueness.append(float(depth_unique / max(n_items, 1)))
    return {
        "n_items": n_items,
        "unique_sids": unique,
        "sid_uniqueness": unique / n_items,
        "sid_entropy_norm": entropy,
        "sid_collision_items": int(n_items - unique),
        "sid_max_multiplicity": int(counts.max()),
        "per_level_unique": per_level_unique,
        "per_level_unique_fraction_codebook": per_level_unique_fraction_codebook,
        "per_level_unique_fraction_items": per_level_unique_fraction_items,
        "per_level_perplexity": per_level_perplexity,
        "per_level_entropy_norm": per_level_entropy,
        "per_depth_unique_sids": per_depth_unique_sids,
        "per_depth_sid_uniqueness": per_depth_sid_uniqueness,
    }


def _progress_bar(iterable: range, enabled: bool):
    if not enabled:
        return iterable
    try:
        from tqdm.std import tqdm
    except ImportError:
        return iterable
    return tqdm(
        iterable,
        desc="RQ-VAE",
        unit="epoch",
        dynamic_ncols=True,
        ascii=True,
        leave=True,
    )


def behavior_alignment_metrics(
    representations: np.ndarray,
    pairs: pd.DataFrame,
    *,
    sample_size: int,
    seed: int,
    top_k: int = 10,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    if len(pairs) > sample_size:
        eval_pairs = pairs.sample(sample_size, random_state=seed)
    else:
        eval_pairs = pairs
    anchors = eval_pairs["item_idx"].to_numpy(np.int64)
    positives = eval_pairs["item_pos"].to_numpy(np.int64)
    reps = representations / np.linalg.norm(representations, axis=1, keepdims=True).clip(1e-8)
    pos_cos = np.sum(reps[anchors] * reps[positives], axis=1)
    random_pos = rng.integers(0, reps.shape[0], size=len(anchors))
    random_cos = np.sum(reps[anchors] * reps[random_pos], axis=1)

    hits = []
    ranks = []
    chunk = 512
    for start in range(0, len(anchors), chunk):
        stop = min(start + chunk, len(anchors))
        score = reps[anchors[start:stop]] @ reps.T
        score[np.arange(stop - start), anchors[start:stop]] = -np.inf
        kth = max(0, min(top_k - 1, reps.shape[0] - 1))
        top = np.argpartition(-score, kth=kth, axis=1)[:, :top_k]
        target = positives[start:stop]
        hit = np.any(top == target[:, None], axis=1)
        hits.extend(hit.tolist())
        sorted_idx = np.argsort(-score, axis=1)
        for row_idx, pos in enumerate(target):
            rank = int(np.where(sorted_idx[row_idx] == pos)[0][0]) + 1
            ranks.append(rank)

    return {
        "positive_cosine_mean": float(np.mean(pos_cos)),
        "random_cosine_mean": float(np.mean(random_cos)),
        "positive_minus_random_cosine": float(np.mean(pos_cos) - np.mean(random_cos)),
        f"behavior_recall_at_{top_k}": float(np.mean(hits)),
        "behavior_median_rank": float(np.median(ranks)),
    }


def evaluate_model(
    model: AdvancedRQVAE,
    bundle: FeatureBundle,
    pairs: pd.DataFrame,
    device: torch.device,
    config: AdvancedRQVAETrainingConfig,
) -> dict[str, Any]:
    sids, contrastive, z_q = encode_all_items(model, bundle, device)
    metrics = sid_metrics(sids, model.config.codebook_sizes)
    metrics.update(
        {
            "contrastive_alignment": behavior_alignment_metrics(
                contrastive,
                pairs,
                sample_size=config.eval_pair_sample,
                seed=config.seed,
                top_k=10,
            ),
            "zq_alignment": behavior_alignment_metrics(
                z_q,
                pairs,
                sample_size=config.eval_pair_sample,
                seed=config.seed,
                top_k=10,
            ),
        }
    )
    return metrics


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_advanced_rqvae_experiment(
    *,
    bundle: FeatureBundle,
    pairs: pd.DataFrame,
    output_dir: str | Path,
    model_config: RQVAEConfig | None = None,
    train_config: AdvancedRQVAETrainingConfig | None = None,
) -> dict[str, Any]:
    train_config = train_config or AdvancedRQVAETrainingConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_config.seed)

    device = torch.device(
        train_config.device if train_config.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model_config = model_config or RQVAEConfig(modality_dims=bundle.modality_dims)
    model = AdvancedRQVAE(model_config).to(device)
    loss_fn = AdvancedRQVAELoss(
        beta=train_config.beta,
        contrastive_weight=train_config.contrastive_weight,
        contrastive_temperature=train_config.contrastive_temperature,
        modality_weights={"meta": 0.75, "description": 1.25},
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )

    item_loader = DataLoader(
        ItemFeatureDataset(bundle),
        batch_size=train_config.item_batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    pair_loader = DataLoader(
        BehaviorPairDataset(bundle, pairs),
        batch_size=train_config.pair_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=train_config.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    item_iter = cycle(item_loader)
    pair_iter = cycle(pair_loader)

    config_payload = {
        "variant": "advanced_rqvae_sid_v2_plum_like",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "n_parameters": count_parameters(model),
        "n_items": bundle.n_items,
        "n_pairs": int(len(pairs)),
    }
    save_json(output_dir / "config.json", config_payload)

    history: dict[str, Any] = {"epochs": [], "eval": []}
    best_score = -float("inf")
    best_epoch = 0
    stopped_early = False
    stop_reason = None

    epoch_bar = _progress_bar(range(1, train_config.epochs + 1), train_config.show_progress)
    for epoch in epoch_bar:
        active_levels_max = _max_active_levels_for_epoch(
            epoch,
            model_config.n_levels,
            train_config,
        )
        active_level_samples: list[int] = []
        model.train()
        epoch_losses: dict[str, list[float]] = {
            "total": [],
            "item_total": [],
            "pair_total": [],
            "recon": [],
            "rq": [],
            "contrastive": [],
        }
        for _ in range(train_config.steps_per_epoch):
            item_batch = _to_device(next(item_iter), device)
            pair_batch = _to_device(next(pair_iter), device)
            active_levels_step = _sample_active_levels(
                active_levels_max,
                model_config.n_levels,
                train_config,
            )
            active_level_samples.append(active_levels_step)

            optimizer.zero_grad(set_to_none=True)
            item_output = model(
                item_batch["inputs"],
                description_mask=item_batch["description_mask"],
                active_levels=active_levels_step,
            )
            item_loss = loss_fn(item_batch["inputs"], item_output)

            anchor_output = model(
                pair_batch["anchor_inputs"],
                description_mask=pair_batch["anchor_description_mask"],
                active_levels=active_levels_step,
            )
            positive_output = model(
                pair_batch["positive_inputs"],
                description_mask=pair_batch["positive_description_mask"],
                active_levels=active_levels_step,
            )
            pair_loss = loss_fn(
                pair_batch["anchor_inputs"],
                anchor_output,
                positive_inputs=pair_batch["positive_inputs"],
                positive_output=positive_output,
                sample_weight=pair_batch["weight"],
            )

            total = 0.5 * item_loss.total + pair_loss.total
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
            optimizer.step()

            epoch_losses["total"].append(float(total.detach().cpu()))
            epoch_losses["item_total"].append(float(item_loss.total.detach().cpu()))
            epoch_losses["pair_total"].append(float(pair_loss.total.detach().cpu()))
            epoch_losses["recon"].append(float(pair_loss.recon.detach().cpu()))
            epoch_losses["rq"].append(float(pair_loss.rq.detach().cpu()))
            epoch_losses["contrastive"].append(float(pair_loss.contrastive.detach().cpu()))

        active_levels_mean = _mean([float(level) for level in active_level_samples])
        epoch_record = {
            "epoch": epoch,
            "active_levels": active_levels_mean,
            "active_levels_max": active_levels_max,
            "active_levels_min": int(min(active_level_samples)),
            "active_levels_last": int(active_level_samples[-1]),
            "active_levels_sampling": train_config.progressive_sampling,
            **{key: _mean(val) for key, val in epoch_losses.items()},
        }

        sids, contrastive, z_q = encode_all_items(
            model,
            bundle,
            device,
            batch_size=train_config.sid_eval_batch_size,
        )
        sid_eval = sid_metrics(sids, model_config.codebook_sizes)
        epoch_record.update(
            {
                "unique_sids": sid_eval["unique_sids"],
                "sid_uniqueness": sid_eval["sid_uniqueness"],
                "sid_entropy_norm": sid_eval["sid_entropy_norm"],
                "sid_collision_items": sid_eval["sid_collision_items"],
                "sid_max_multiplicity": sid_eval["sid_max_multiplicity"],
                "per_level_unique": sid_eval["per_level_unique"],
                "per_level_unique_fraction_codebook": sid_eval[
                    "per_level_unique_fraction_codebook"
                ],
                "per_level_unique_fraction_items": sid_eval[
                    "per_level_unique_fraction_items"
                ],
                "per_level_perplexity": sid_eval["per_level_perplexity"],
                "per_level_entropy_norm": sid_eval["per_level_entropy_norm"],
                "per_depth_unique_sids": sid_eval["per_depth_unique_sids"],
                "per_depth_sid_uniqueness": sid_eval["per_depth_sid_uniqueness"],
            }
        )
        history["epochs"].append(epoch_record)

        score = float(sid_eval["sid_uniqueness"])
        improved = score > best_score + train_config.early_stopping_min_delta
        should_eval = epoch == 1 or epoch % train_config.eval_every == 0 or epoch == train_config.epochs
        if should_eval:
            eval_metrics = {
                **sid_eval,
                "contrastive_alignment": behavior_alignment_metrics(
                    contrastive,
                    pairs,
                    sample_size=train_config.eval_pair_sample,
                    seed=train_config.seed,
                    top_k=10,
                ),
                "zq_alignment": behavior_alignment_metrics(
                    z_q,
                    pairs,
                    sample_size=train_config.eval_pair_sample,
                    seed=train_config.seed,
                    top_k=10,
                ),
            }
            eval_record = {"epoch": epoch, **eval_metrics}
            history["eval"].append(eval_record)

        if improved:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_config": asdict(model_config),
                    "train_config": asdict(train_config),
                    "sid_eval": {"epoch": epoch, **sid_eval},
                },
                output_dir / "checkpoint_best.pt",
            )

        save_json(output_dir / "metrics.json", history)

        if hasattr(epoch_bar, "set_postfix"):
            sampling_mode = train_config.progressive_sampling.lower().strip()
            epoch_bar.set_postfix(
                {
                    "loss": f"{epoch_record['total']:.4f}",
                    "levels": (
                        f"{active_levels_mean:.2f}/{active_levels_max}"
                        if sampling_mode == "uniform_prefix"
                        else active_levels_max
                    ),
                    "sid": f"{score:.4f}",
                    "best_sid": f"{best_score:.4f}",
                    "bad_epochs": epoch - best_epoch,
                }
            )

        patience = train_config.early_stopping_patience
        if patience is not None and epoch - best_epoch >= patience:
            stopped_early = True
            stop_reason = (
                f"sid_uniqueness did not improve for {patience} consecutive epochs"
            )
            break

    torch.save(
        {
            "model": model.state_dict(),
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
        },
        output_dir / "checkpoint_last.pt",
    )

    final_eval = evaluate_model(model, bundle, pairs, device, train_config)
    sids, _, _ = encode_all_items(model, bundle, device)
    np.save(output_dir / "SIDs.npy", sids.astype(np.int64))
    pd.DataFrame(
        {
            "item_idx": bundle.item_idx,
            "movie_id": bundle.movie_id,
            **{f"sid_{level}": sids[:, level] for level in range(sids.shape[1])},
        }
    ).to_parquet(output_dir / "sid_mapping.parquet", index=False)

    summary = {
        "output_dir": str(output_dir),
        "best_epoch": best_epoch,
        "best_sid_uniqueness": best_score,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "final_eval": final_eval,
        "n_parameters": count_parameters(model),
    }
    save_json(output_dir / "summary.json", summary)
    return summary
