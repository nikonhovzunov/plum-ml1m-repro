"""Run the advanced PLUM-like RQ-VAE/SID-v2 experiment.

This script builds multi-modal movie text features, mines weighted behavioral
co-occurrence positives, and trains the advanced RQ-VAE model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sid import (  # noqa: E402
    AdvancedRQVAETrainingConfig,
    RQVAEConfig,
    build_item_text_profiles,
    build_weighted_cooccurrence_pairs,
    encode_text_modalities,
    load_feature_bundle,
    run_advanced_rqvae_experiment,
)


def parse_codebook_sizes(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def safe_model_slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_").replace(".", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--no-local-files-only", action="store_true")
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--rebuild-pairs", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--steps-per-epoch", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--item-batch-size", type=int, default=512)
    parser.add_argument("--pair-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--branch-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--codebook-sizes", default="2048,1024,512,256")
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--distance-decay", type=float, default=0.85)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--top-k-per-item", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        default="runs/advanced_rqvae_sid_v2_bge_large_ppmi",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_slug = safe_model_slug(args.embedding_model)

    item_meta_path = ROOT / "data/processed/item_features/item_meta.parquet"
    overviews_path = ROOT / "research/movie_overviews/data/ml1m_movie_overviews.csv"
    train_path = ROOT / "data/processed/splits/train.parquet"
    feature_path = ROOT / f"data/processed/item_features/rqvae_v2_{model_slug}_meta_desc.npz"
    profile_path = ROOT / f"data/processed/item_features/rqvae_v2_{model_slug}_profiles.parquet"
    pair_path = (
        ROOT
        / f"data/processed/sid_pairs/co_pairs_behavior_ppmi_w{args.window_size}_r{int(args.min_rating)}_top{args.top_k_per_item}.parquet"
    )
    output_dir = ROOT / args.output_dir

    if args.rebuild_features or not feature_path.exists():
        item_meta = pd.read_parquet(item_meta_path)
        overviews = pd.read_csv(overviews_path)
        profiles = build_item_text_profiles(item_meta, overviews)
        profiles.to_parquet(profile_path, index=False)
        bundle = encode_text_modalities(
            profiles,
            output_path=feature_path,
            model_name=args.embedding_model,
            batch_size=64,
            device="cuda",
            local_files_only=not args.no_local_files_only,
        )
    else:
        bundle = load_feature_bundle(feature_path)

    if args.rebuild_pairs or not pair_path.exists():
        train = pd.read_parquet(train_path)
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
        codebook_sizes=parse_codebook_sizes(args.codebook_sizes),
        dropout=args.dropout,
    )
    train_config = AdvancedRQVAETrainingConfig(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        eval_every=args.eval_every,
        item_batch_size=args.item_batch_size,
        pair_batch_size=args.pair_batch_size,
        lr=args.lr,
        contrastive_weight=args.contrastive_weight,
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
