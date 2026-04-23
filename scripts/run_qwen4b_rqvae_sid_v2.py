"""Train PLUM-style RQ-VAE Semantic IDs on Qwen4B item embeddings."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sid import (  # noqa: E402
    AdvancedRQVAE,
    AdvancedRQVAETrainingConfig,
    RQVAEConfig,
    build_weighted_cooccurrence_pairs,
    count_parameters,
    load_feature_bundle,
    run_advanced_rqvae_experiment,
)


def parse_codebook_sizes(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def parse_progressive_schedule(raw: str) -> tuple[tuple[int, int], ...] | None:
    raw = raw.strip()
    if not raw:
        return None
    schedule: list[tuple[int, int]] = []
    for part in raw.split(","):
        epoch, levels = part.split(":")
        schedule.append((int(epoch.strip()), int(levels.strip())))
    return tuple(schedule)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-path",
        default="data/processed/item_features/qwen4b_audited_v1_meta_desc_embeddings.npz",
    )
    parser.add_argument(
        "--pair-path",
        default="data/processed/sid_pairs/co_pairs_behavior_ppmi_w20_r4_top32.parquet",
    )
    parser.add_argument("--rebuild-pairs", action="store_true")
    parser.add_argument("--output-dir", default="runs/qwen4b_rqvae_sid_v2_plum")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--steps-per-epoch", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-pair-sample", type=int, default=10000)
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument("--item-batch-size", type=int, default=512)
    parser.add_argument("--pair-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--progressive-warmup-epochs", type=int, default=24)
    parser.add_argument(
        "--progressive-schedule",
        default="",
        help="Optional deterministic schedule like '1:1,6:2,12:3,18:4'.",
    )
    parser.add_argument(
        "--progressive-sampling",
        default="deterministic",
        choices=["deterministic", "uniform_prefix"],
        help="Use 'uniform_prefix' to sample active depth from 1..max depth each step.",
    )
    parser.add_argument("--latent-dim", type=int, default=320)
    parser.add_argument("--branch-dim", type=int, default=320)
    parser.add_argument("--hidden-dims", default="640,640")
    parser.add_argument("--decoder-hidden-dims", default="640,640")
    parser.add_argument("--codebook-sizes", default="1024,512,256,128")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--max-params", type=int, default=15_000_000)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--distance-decay", type=float, default=0.85)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--top-k-per-item", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    feature_path = ROOT / args.feature_path
    pair_path = ROOT / args.pair_path
    output_dir = ROOT / args.output_dir

    bundle = load_feature_bundle(feature_path)

    if args.rebuild_pairs or not pair_path.exists():
        train = pd.read_parquet(ROOT / "data/processed/splits/train.parquet")
        pairs = build_weighted_cooccurrence_pairs(
            train,
            output_path=pair_path,
            window_size=args.window_size,
            distance_decay=args.distance_decay,
            min_rating=args.min_rating,
            top_k_per_item=args.top_k_per_item,
        )
    else:
        pairs = pd.read_parquet(pair_path)

    model_config = RQVAEConfig(
        modality_dims=bundle.modality_dims,
        latent_dim=args.latent_dim,
        branch_dim=args.branch_dim,
        branch_hidden_dims=parse_ints(args.hidden_dims),
        fusion_hidden_dims=parse_ints(args.hidden_dims),
        decoder_hidden_dims=parse_ints(args.decoder_hidden_dims),
        codebook_sizes=parse_codebook_sizes(args.codebook_sizes),
        dropout=args.dropout,
        use_description_mask=False,
        contrastive_dim=128,
    )
    n_params = count_parameters(AdvancedRQVAE(model_config))
    if n_params >= args.max_params:
        raise ValueError(f"RQ-VAE has {n_params:,} parameters; max is {args.max_params:,}.")

    train_config = AdvancedRQVAETrainingConfig(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        eval_every=args.eval_every,
        item_batch_size=args.item_batch_size,
        pair_batch_size=args.pair_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        contrastive_weight=args.contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,
        progressive_masking=True,
        progressive_warmup_epochs=args.progressive_warmup_epochs,
        progressive_schedule=parse_progressive_schedule(args.progressive_schedule),
        progressive_sampling=args.progressive_sampling,
        eval_pair_sample=args.eval_pair_sample,
        early_stopping_patience=args.early_stopping_patience,
        show_progress=True,
    )

    summary = run_advanced_rqvae_experiment(
        bundle=bundle,
        pairs=pairs,
        output_dir=output_dir,
        model_config=model_config,
        train_config=train_config,
    )
    print(summary)


if __name__ == "__main__":
    main()
