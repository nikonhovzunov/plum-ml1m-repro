from __future__ import annotations

import base64
import json
import math
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = ROOT / "notebooks" / "reporting" / "07_experiments_report.ipynb"
FIG_DIR = ROOT / "reports" / "status" / "figures" / "experiments_report_2026-04-21"
SUMMARY_PATH = ROOT / "reports" / "status" / "experiments_report_2026-04-21_summary.json"

SPLIT_DIR = ROOT / "data" / "processed" / "splits"
SID_PATH = ROOT / "data" / "processed" / "artifacts" / "CPT_user_behavior_V1" / "SIDs_V1.npy"
SFT_DIR = ROOT / "data" / "processed" / "artifacts" / "SFT"
CPT_DIR = ROOT / "data" / "processed" / "artifacts" / "CPT_user_behavior_V1"
RUNS_DIR = ROOT / "runs"


plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 180,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_values(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    out: list[float] = []
    for value in values:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out.append(float(value))
    return out


def last_finite(values: Any) -> float | None:
    vals = finite_values(values)
    return vals[-1] if vals else None


def min_finite(values: Any) -> float | None:
    vals = finite_values(values)
    return min(vals) if vals else None


def max_finite(values: Any) -> float | None:
    vals = finite_values(values)
    return max(vals) if vals else None


def pct(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{100 * float(value):.2f}%"


def safe_round(value: Any, ndigits: int = 4) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return round(float(value), ndigits)
    return value


def as_html_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    shown = df.head(max_rows).copy()
    return shown.to_html(index=False, escape=False, classes="dataframe compact")


def table_output(df: pd.DataFrame, max_rows: int = 40) -> dict[str, Any]:
    shown = df.head(max_rows)
    suffix = ""
    if len(df) > max_rows:
        suffix = f"\n... показано {max_rows} из {len(df)} строк"
    return {
        "output_type": "execute_result",
        "execution_count": None,
        "metadata": {},
        "data": {
            "text/plain": shown.to_string(index=False) + suffix,
            "text/html": as_html_table(df, max_rows=max_rows),
        },
    }


def text_output(text: str) -> dict[str, Any]:
    return {"output_type": "stream", "name": "stdout", "text": text if text.endswith("\n") else text + "\n"}


def image_output(path: Path) -> dict[str, Any]:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "output_type": "display_data",
        "metadata": {},
        "data": {
            "image/png": encoded,
            "text/plain": f"<Figure: {path.name}>",
        },
    }


def markdown_cell(text: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(source: str, outputs: list[dict[str, Any]] | None = None, execution_count: int | None = None) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": outputs or [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


def recall_at_k(candidates: list[int], target: int, k: int) -> float:
    return float(int(int(target) in candidates[:k]))


def ndcg_at_k(candidates: list[int], target: int, k: int) -> float:
    target = int(target)
    for rank, item in enumerate(candidates[:k], start=1):
        if int(item) == target:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def mrr_at_k(candidates: list[int], target: int, k: int) -> float:
    target = int(target)
    for rank, item in enumerate(candidates[:k], start=1):
        if int(item) == target:
            return 1.0 / rank
    return 0.0


def popularity_metrics(
    target_df: pd.DataFrame,
    train_like_df: pd.DataFrame,
    popularity_items: list[int],
    k_values: tuple[int, ...] = (1, 5, 10, 20),
) -> dict[str, float]:
    seen_by_user = {
        int(user_id): set(map(int, group["item_idx"]))
        for user_id, group in train_like_df.groupby("user_id", sort=False)
    }
    rows = list(target_df.sort_values(["user_idx", "timestamp"]).itertuples())
    metrics: dict[str, float] = {"n": float(len(rows))}

    for k in k_values:
        recall = ndcg = mrr = 0.0
        for row in rows:
            seen = seen_by_user.get(int(row.user_id), set())
            candidates: list[int] = []
            for item in popularity_items:
                if item not in seen:
                    candidates.append(int(item))
                    if len(candidates) >= k:
                        break
            target = int(row.item_idx)
            recall += recall_at_k(candidates, target, k)
            ndcg += ndcg_at_k(candidates, target, k)
            mrr += mrr_at_k(candidates, target, k)
        denom = max(len(rows), 1)
        metrics[f"recall@{k}"] = recall / denom
        metrics[f"ndcg@{k}"] = ndcg / denom
        metrics[f"mrr@{k}"] = mrr / denom

    metrics["item_coverage"] = float(min(max(k_values), len(popularity_items)))
    return metrics


def evaluate_prediction_records(path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics: dict[str, Any] = {"n": float(len(records))}
    for k in (1, 5, 10, 20):
        recall = ndcg = mrr = 0.0
        for record in records:
            candidates = [int(item) for item in record.get("candidates", [])]
            target = int(record["target_item_idx"])
            recall += recall_at_k(candidates, target, k)
            ndcg += ndcg_at_k(candidates, target, k)
            mrr += mrr_at_k(candidates, target, k)
        denom = max(len(records), 1)
        metrics[f"recall@{k}"] = recall / denom
        metrics[f"ndcg@{k}"] = ndcg / denom
        metrics[f"mrr@{k}"] = mrr / denom

    lengths = [len(record.get("candidates", [])) for record in records]
    top1 = [record["candidates"][0] for record in records if record.get("candidates")]
    top1_counts = Counter(top1)
    all_candidates = [item for record in records for item in record.get("candidates", [])]
    metrics["candidate_len_mean"] = float(np.mean(lengths)) if lengths else 0.0
    metrics["candidate_len_min"] = int(np.min(lengths)) if lengths else 0
    metrics["candidate_len_lt_10"] = int(sum(length < 10 for length in lengths))
    metrics["candidate_len_lt_20"] = int(sum(length < 20 for length in lengths))
    metrics["top1_unique"] = float(len(top1_counts))
    metrics["top1_top10_share"] = (
        float(sum(count for _, count in top1_counts.most_common(10)) / len(top1)) if top1 else 0.0
    )
    metrics["item_coverage"] = float(len(set(all_candidates)))
    return metrics


def load_dataset_bundle() -> dict[str, Any]:
    train = pd.read_parquet(SPLIT_DIR / "train.parquet")
    val = pd.read_parquet(SPLIT_DIR / "val.parquet")
    test = pd.read_parquet(SPLIT_DIR / "test.parquet")
    sids = np.load(SID_PATH)

    sid_tuples = [tuple(map(int, sid)) for sid in sids.tolist()]
    sid_counts = Counter(sid_tuples)
    history_lengths = train.groupby("user_id").size()
    popularity_items = train["item_idx"].value_counts().index.astype(int).tolist()

    val_sorted = val.sort_values(["user_idx", "timestamp"]).reset_index(drop=True)
    test_sorted = test.sort_values(["user_idx", "timestamp"]).reset_index(drop=True)

    popularity = {
        "popularity_full_val": popularity_metrics(val_sorted, train, popularity_items),
        "popularity_val_probe_1024": popularity_metrics(val_sorted.head(1024), train, popularity_items),
        "popularity_full_test": popularity_metrics(test_sorted, pd.concat([train, val], ignore_index=True), popularity_items),
    }

    target_distribution: list[dict[str, Any]] = []
    for split_name, frame in [("train", train), ("val", val), ("test", test)]:
        vc = frame["item_idx"].astype(int).value_counts()
        target_distribution.append(
            {
                "split": split_name,
                "n": len(frame),
                "unique_items": int(vc.size),
                "top10_share": float(vc.head(10).sum() / max(len(frame), 1)),
                "top20_share": float(vc.head(20).sum() / max(len(frame), 1)),
                "top100_share": float(vc.head(100).sum() / max(len(frame), 1)),
            }
        )

    split_summary = pd.DataFrame(
        [
            {
                "split": "train",
                "rows": len(train),
                "users": train["user_id"].nunique(),
                "items": train["item_idx"].nunique(),
            },
            {
                "split": "val",
                "rows": len(val),
                "users": val["user_id"].nunique(),
                "items": val["item_idx"].nunique(),
            },
            {
                "split": "test",
                "rows": len(test),
                "users": test["user_id"].nunique(),
                "items": test["item_idx"].nunique(),
            },
        ]
    )

    dataset_facts = {
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "users": int(train["user_id"].nunique()),
        "items": int(len(sids)),
        "unique_sids": int(len(sid_counts)),
        "sid_uniqueness": float(len(sid_counts) / len(sids)),
        "sid_collision_buckets": int(sum(1 for count in sid_counts.values() if count > 1)),
        "sid_collided_items": int(sum(count for count in sid_counts.values() if count > 1)),
        "sid_collision_excess": int(len(sids) - len(sid_counts)),
        "history_len_median": float(history_lengths.median()),
        "history_len_mean": float(history_lengths.mean()),
        "history_len_p90": float(history_lengths.quantile(0.9)),
        "history_len_max": int(history_lengths.max()),
    }

    return {
        "train": train,
        "val": val,
        "test": test,
        "sids": sids,
        "history_lengths": history_lengths,
        "split_summary": split_summary,
        "dataset_facts": dataset_facts,
        "popularity": popularity,
        "target_distribution": pd.DataFrame(target_distribution),
    }


def collect_rqvae_runs() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(RUNS_DIR.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        try:
            metrics = read_json(metrics_path)
        except Exception:
            continue

        config_path = run_dir / "config.json"
        config = read_json(config_path) if config_path.exists() else {}
        val_loss = finite_values(metrics.get("val_loss"))
        sid_uniqueness = finite_values(metrics.get("sid_uniqueness_all"))
        train_loss = finite_values(metrics.get("train_loss"))
        rows.append(
            {
                "run": run_dir.name,
                "n_levels": config.get("n_levels"),
                "codebook_sizes": str(config.get("codebook_sizes", "")),
                "epochs_logged": max((len(v) for v in metrics.values() if isinstance(v, list)), default=0),
                "best_val_loss": min(val_loss) if val_loss else None,
                "last_val_loss": val_loss[-1] if val_loss else None,
                "last_train_loss": train_loss[-1] if train_loss else None,
                "best_sid_uniqueness_pct": max(sid_uniqueness) if sid_uniqueness else None,
                "last_sid_uniqueness_pct": sid_uniqueness[-1] if sid_uniqueness else None,
                "has_contrastive": "train_con" in metrics,
                "path": str(run_dir.relative_to(ROOT)),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["quality_note"] = np.where(
        df["best_sid_uniqueness_pct"].fillna(0) >= 90,
        "candidate SID export",
        np.where(df["best_sid_uniqueness_pct"].fillna(0) >= 80, "strong SID run", "diagnostic/weak"),
    )
    return df.sort_values(["best_sid_uniqueness_pct", "best_val_loss"], ascending=[False, True], na_position="last")


def trainer_state_rows(base_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state_path in sorted(base_dir.glob("cpt_*/checkpoint-*/trainer_state.json")):
        state = read_json(state_path)
        log_history = state.get("log_history", [])
        eval_losses = [entry["eval_loss"] for entry in log_history if "eval_loss" in entry and math.isfinite(float(entry["eval_loss"]))]
        train_losses = [entry["loss"] for entry in log_history if "loss" in entry and math.isfinite(float(entry["loss"]))]
        run = state_path.parents[1].name
        checkpoint = state_path.parent.name
        rows.append(
            {
                "run": run,
                "checkpoint": checkpoint,
                "family": "large" if "large" in run else ("medium" if "medium" in run else "base"),
                "global_step": state.get("global_step"),
                "epoch": state.get("epoch"),
                "best_eval_loss": min(eval_losses) if eval_losses else state.get("best_metric"),
                "last_eval_loss": eval_losses[-1] if eval_losses else None,
                "last_train_loss": train_losses[-1] if train_losses else None,
                "best_model_checkpoint": state.get("best_model_checkpoint"),
                "path": str(state_path.parent.relative_to(ROOT)),
            }
        )
    return rows


def collect_cpt_runs() -> pd.DataFrame:
    df = pd.DataFrame(trainer_state_rows(CPT_DIR))
    if df.empty:
        return df
    return df.sort_values(["family", "run", "global_step"])


def collect_sft_runs(bundle: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for split_name, metrics in bundle["popularity"].items():
        metric_rows.append(
            {
                "experiment": split_name,
                "stage": "baseline",
                "scope": split_name.replace("popularity_", ""),
                **{key: metrics.get(key) for key in ["n", "recall@1", "recall@5", "recall@10", "recall@20", "ndcg@10", "ndcg@20", "mrr@10", "mrr@20", "item_coverage"]},
            }
        )

    weak_summary_path = SFT_DIR / "sft_gpt2_s_weak_cpt_3epochs" / "run_summary.json"
    if weak_summary_path.exists():
        summary = read_json(weak_summary_path)
        for split in ("val", "test"):
            metrics = summary.get(f"{split}_metrics", {})
            metric_rows.append(
                {
                    "experiment": "sft_gpt2_s_weak_cpt_3epochs",
                    "stage": "SFT",
                    "scope": f"full_{split}",
                    **{key: metrics.get(key) for key in ["n", "recall@1", "recall@5", "recall@10", "recall@20", "ndcg@10", "ndcg@20", "mrr@10", "mrr@20", "item_coverage"]},
                }
            )
            diagnostic_rows.append(
                {
                    "experiment": "sft_gpt2_s_weak_cpt_3epochs",
                    "scope": f"full_{split}",
                    "invalid_sid_rate": metrics.get("invalid_sid_rate"),
                    "duplicate_rate": metrics.get("duplicate_rate"),
                    "item_coverage": metrics.get("item_coverage"),
                    "decode_seconds": metrics.get("decode_seconds"),
                    "candidate_len_mean": None,
                    "candidate_len_lt_20": None,
                    "top1_unique": None,
                    "top1_top10_share": None,
                }
            )

    medium_dir = SFT_DIR / "sft_gpt2_medium_plus_plus_plus_monitor"
    epoch_metrics_path = medium_dir / "logs" / "epoch_metrics.json"
    if epoch_metrics_path.exists():
        for row in read_json(epoch_metrics_path):
            metric_rows.append(
                {
                    "experiment": "sft_gpt2_medium_plus_plus_plus_monitor",
                    "stage": "SFT",
                    "scope": f"val_probe_epoch_{int(row['epoch']):02d}",
                    **{key: row.get(key) for key in ["n", "recall@1", "recall@5", "recall@10", "recall@20", "ndcg@10", "ndcg@20", "mrr@10", "mrr@20", "item_coverage"]},
                }
            )

        for pred_path in sorted((medium_dir / "predictions").glob("epoch-*_val_probe_predictions.jsonl")):
            epoch = pred_path.name.split("_")[0].replace("epoch-", "")
            diagnostics = evaluate_prediction_records(pred_path)
            diagnostic_rows.append(
                {
                    "experiment": "sft_gpt2_medium_plus_plus_plus_monitor",
                    "scope": f"val_probe_epoch_{epoch}",
                    "invalid_sid_rate": 0.0,
                    "duplicate_rate": 0.0,
                    "item_coverage": diagnostics.get("item_coverage"),
                    "decode_seconds": None,
                    "candidate_len_mean": diagnostics.get("candidate_len_mean"),
                    "candidate_len_lt_20": diagnostics.get("candidate_len_lt_20"),
                    "top1_unique": diagnostics.get("top1_unique"),
                    "top1_top10_share": diagnostics.get("top1_top10_share"),
                }
            )

    tiny_state_path = SFT_DIR / "sft_tiny_gpt2_smoke_100steps" / "checkpoint-100" / "trainer_state.json"
    if tiny_state_path.exists():
        state = read_json(tiny_state_path)
        eval_losses = [entry["eval_loss"] for entry in state.get("log_history", []) if "eval_loss" in entry]
        train_losses = [entry["loss"] for entry in state.get("log_history", []) if "loss" in entry]
        diagnostic_rows.append(
            {
                "experiment": "sft_tiny_gpt2_smoke_100steps",
                "scope": "smoke",
                "invalid_sid_rate": None,
                "duplicate_rate": None,
                "item_coverage": None,
                "decode_seconds": None,
                "candidate_len_mean": None,
                "candidate_len_lt_20": None,
                "top1_unique": None,
                "top1_top10_share": None,
                "best_eval_loss": min(eval_losses) if eval_losses else None,
                "last_train_loss": train_losses[-1] if train_losses else None,
                "global_step": state.get("global_step"),
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    diagnostics_df = pd.DataFrame(diagnostic_rows)
    return metrics_df, diagnostics_df


def plot_dataset(bundle: dict[str, Any]) -> Path:
    path = FIG_DIR / "01_dataset_history.png"
    history = bundle["history_lengths"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(bundle["split_summary"]["split"], bundle["split_summary"]["rows"], color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0].set_title("Split sizes")
    axes[0].set_ylabel("interactions")
    axes[0].ticklabel_format(axis="y", style="plain")

    axes[1].hist(history.clip(upper=600), bins=40, color="#4C78A8", alpha=0.85)
    axes[1].axvline(history.median(), color="#E45756", linestyle="--", label=f"median={history.median():.0f}")
    axes[1].axvline(100, color="#72B7B2", linestyle="--", label="history cap=100")
    axes[1].set_title("Train history length per user, clipped at 600")
    axes[1].set_xlabel("events")
    axes[1].set_ylabel("users")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_rqvae(rqvae_df: pd.DataFrame, exported_sid_uniqueness: float) -> Path:
    path = FIG_DIR / "02_rqvae_uniqueness.png"
    plot_df = rqvae_df.dropna(subset=["best_sid_uniqueness_pct"]).copy()
    plot_df = plot_df.sort_values("best_sid_uniqueness_pct", ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.barh(plot_df["run"], plot_df["best_sid_uniqueness_pct"], color="#4C78A8")
    ax.axvline(exported_sid_uniqueness * 100, color="#E45756", linestyle="--", label=f"exported SIDs={exported_sid_uniqueness * 100:.2f}%")
    ax.set_title("RQ-VAE runs: best observed full-SID uniqueness")
    ax.set_xlabel("unique SID tuples, %")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_cpt(cpt_df: pd.DataFrame) -> Path:
    path = FIG_DIR / "03_cpt_eval_loss.png"
    if cpt_df.empty:
        return path
    plot_df = cpt_df.dropna(subset=["best_eval_loss"]).sort_values("best_eval_loss").head(18).copy()
    plot_df["label"] = plot_df["run"] + "/" + plot_df["checkpoint"].astype(str)
    colors = plot_df["family"].map({"base": "#4C78A8", "medium": "#F58518", "large": "#54A24B"}).fillna("#888888")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(plot_df["label"][::-1], plot_df["best_eval_loss"][::-1], color=list(colors[::-1]))
    ax.set_title("CPT checkpoints: best eval loss in Trainer state")
    ax.set_xlabel("best eval loss")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_sft_metrics(sft_metrics: pd.DataFrame) -> Path:
    path = FIG_DIR / "04_sft_vs_baselines.png"
    scopes = [
        "val_probe_1024",
        "val_probe_epoch_03",
        "full_val",
        "full_test",
    ]
    labels = {
        "popularity_val_probe_1024": "Popularity val probe",
        "sft_gpt2_medium_plus_plus_plus_monitor::val_probe_epoch_03": "Medium SFT val probe",
        "popularity_full_val": "Popularity full val",
        "sft_gpt2_s_weak_cpt_3epochs::full_val": "GPT2-S SFT full val",
        "popularity_full_test": "Popularity full test",
        "sft_gpt2_s_weak_cpt_3epochs::full_test": "GPT2-S SFT full test",
    }

    rows: list[dict[str, Any]] = []
    for _, row in sft_metrics.iterrows():
        key = row["experiment"] if row["stage"] == "baseline" else f"{row['experiment']}::{row['scope']}"
        if key in labels:
            rows.append(
                {
                    "label": labels[key],
                    "recall@10": row.get("recall@10"),
                    "recall@20": row.get("recall@20"),
                    "ndcg@10": row.get("ndcg@10"),
                }
            )
    plot_df = pd.DataFrame(rows).drop_duplicates("label")
    order = [
        "Popularity val probe",
        "Medium SFT val probe",
        "Popularity full val",
        "GPT2-S SFT full val",
        "Popularity full test",
        "GPT2-S SFT full test",
    ]
    plot_df["label"] = pd.Categorical(plot_df["label"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("label")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.36
    x = np.arange(len(plot_df))
    axes[0].bar(x - width / 2, plot_df["recall@10"], width, label="Recall@10", color="#4C78A8")
    axes[0].bar(x + width / 2, plot_df["recall@20"], width, label="Recall@20", color="#F58518")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(plot_df["label"], rotation=25, ha="right")
    axes[0].set_title("Ranking metrics: SFT vs popularity")
    axes[0].set_ylabel("metric")
    axes[0].legend()

    axes[1].bar(plot_df["label"], plot_df["ndcg@10"], color="#54A24B")
    axes[1].tick_params(axis="x", labelrotation=25)
    for label in axes[1].get_xticklabels():
        label.set_horizontalalignment("right")
    axes[1].set_title("NDCG@10")
    axes[1].set_ylabel("metric")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_sft_epochs(medium_epoch_df: pd.DataFrame) -> Path:
    path = FIG_DIR / "05_medium_sft_epochs.png"
    if medium_epoch_df.empty:
        return path
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(medium_epoch_df["epoch"], medium_epoch_df["teacher_forcing_eval_loss"], marker="o", color="#4C78A8")
    axes[0].set_title("Teacher-forcing eval loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[1].plot(medium_epoch_df["epoch"], medium_epoch_df["recall@10"], marker="o", label="Recall@10", color="#F58518")
    axes[1].plot(medium_epoch_df["epoch"], medium_epoch_df["ndcg@10"], marker="o", label="NDCG@10", color="#54A24B")
    axes[1].set_title("Val-probe recsys metrics")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[2].plot(medium_epoch_df["epoch"], medium_epoch_df["item_coverage"], marker="o", color="#B279A2")
    axes[2].set_title("Candidate item coverage")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("unique items")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_sft_diagnostics(sft_diag: pd.DataFrame) -> Path:
    path = FIG_DIR / "06_sft_diagnostics.png"
    plot_df = sft_diag[sft_diag["experiment"].eq("sft_gpt2_medium_plus_plus_plus_monitor")].copy()
    if plot_df.empty:
        return path
    plot_df["epoch"] = plot_df["scope"].str.extract(r"(\d+)$").astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(plot_df["epoch"], plot_df["candidate_len_mean"], marker="o", label="mean len")
    axes[0].bar(plot_df["epoch"], plot_df["candidate_len_lt_20"], alpha=0.35, label="rows with <20 candidates")
    axes[0].set_title("Generated candidate list health")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(plot_df["epoch"], plot_df["top1_unique"], marker="o", label="unique top1")
    axes[1].plot(plot_df["epoch"], plot_df["item_coverage"], marker="o", label="topK coverage")
    axes[1].set_title("Recommendation diversity")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def create_notebook(
    bundle: dict[str, Any],
    rqvae_df: pd.DataFrame,
    cpt_df: pd.DataFrame,
    sft_metrics: pd.DataFrame,
    sft_diag: pd.DataFrame,
    figures: dict[str, Path],
) -> None:
    facts = bundle["dataset_facts"]
    best_rqvae = rqvae_df.iloc[0].to_dict() if not rqvae_df.empty else {}
    best_cpt = cpt_df.dropna(subset=["best_eval_loss"]).sort_values("best_eval_loss").head(1)
    best_cpt_row = best_cpt.iloc[0].to_dict() if not best_cpt.empty else {}

    medium_epoch_path = SFT_DIR / "sft_gpt2_medium_plus_plus_plus_monitor" / "logs" / "epoch_metrics.json"
    medium_epochs = pd.DataFrame(read_json(medium_epoch_path)) if medium_epoch_path.exists() else pd.DataFrame()
    medium_best = medium_epochs.sort_values("recall@10", ascending=False).head(1)
    medium_best_row = medium_best.iloc[0].to_dict() if not medium_best.empty else {}

    metric_view_cols = [
        "experiment",
        "stage",
        "scope",
        "n",
        "recall@10",
        "recall@20",
        "ndcg@10",
        "ndcg@20",
        "item_coverage",
    ]
    metric_view = sft_metrics[[col for col in metric_view_cols if col in sft_metrics.columns]].copy()
    for col in ["recall@10", "recall@20", "ndcg@10", "ndcg@20"]:
        if col in metric_view:
            metric_view[col] = metric_view[col].map(lambda x: safe_round(x, 4))

    rqvae_view = rqvae_df[
        [
            "run",
            "n_levels",
            "epochs_logged",
            "best_val_loss",
            "best_sid_uniqueness_pct",
            "last_sid_uniqueness_pct",
            "quality_note",
            "path",
        ]
    ].copy()
    for col in ["best_val_loss", "best_sid_uniqueness_pct", "last_sid_uniqueness_pct"]:
        rqvae_view[col] = rqvae_view[col].map(lambda x: safe_round(x, 4))

    cpt_view = cpt_df[
        [
            "run",
            "checkpoint",
            "family",
            "global_step",
            "epoch",
            "best_eval_loss",
            "last_train_loss",
            "path",
        ]
    ].copy()
    for col in ["epoch", "best_eval_loss", "last_train_loss"]:
        cpt_view[col] = cpt_view[col].map(lambda x: safe_round(x, 4))

    diag_view = sft_diag.copy()
    for col in diag_view.columns:
        if col not in {"experiment", "scope"}:
            diag_view[col] = diag_view[col].map(lambda x: safe_round(x, 4))

    cells: list[dict[str, Any]] = []
    cells.append(
        markdown_cell(
            f"""# Отчет по экспериментам PLUM-style MovieLens-1M

Дата сборки: `{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}`  
Репозиторий: `{ROOT}`

Этот notebook создан и выполнен автоматически из локальных артефактов проекта: `runs/`, `data/processed/artifacts/CPT_user_behavior_V1/`, `data/processed/artifacts/SFT/`, `data/processed/splits/`.

Цель отчета: честно зафиксировать, что получилось на стадиях **Semantic ID / RQ-VAE**, **CPT**, **SFT**, где пайплайн рабочий, а где текущая серия экспериментов не проходит минимальный baseline-контроль.
"""
        )
    )
    cells.append(
        markdown_cell(
            f"""## Executive Summary

1. **Самая сильная часть проекта сейчас - Semantic ID stage.** Экспортированные `SIDs_V1` имеют `{facts["unique_sids"]}` уникальных SID tuple из `{facts["items"]}` items: **{pct(facts["sid_uniqueness"])}**, collision excess `{facts["sid_collision_excess"]}`.
2. **CPT stage технически выполнен на нескольких GPT-2/GPT-2 Medium ветках.** Лучший найденный CPT eval loss по trainer states: `{safe_round(best_cpt_row.get("best_eval_loss"), 4)}` у `{best_cpt_row.get("run", "n/a")}/{best_cpt_row.get("checkpoint", "n/a")}`. Это полезно как domain adaptation, но само по себе не доказывает recommendation quality.
3. **SFT stage работает технически, но текущая серия является negative result.** Medium SFT на val-probe дает лучший `Recall@10 = {safe_round(medium_best_row.get("recall@10"), 4)}`, тогда как простой popularity baseline на том же probe дает `Recall@10 = {safe_round(bundle["popularity"]["popularity_val_probe_1024"]["recall@10"], 4)}`.
4. **Продолжать эту же серию без смены постановки нерационально.** Следующий осмысленный шаг: зафиксировать negative result, добавить сильные sequential baselines и только потом запускать новую SFT-серию с full sliding-window supervision, `MAX_LENGTH=1024`, честным `RECSYS_BEAMS >= 20` и контролем diversity.
"""
        )
    )
    cells.append(
        code_cell(
            """# Re-runnable setup for this report notebook.
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from IPython.display import Image, display

NOTEBOOK_ROOT = Path.cwd().resolve()
while not (NOTEBOOK_ROOT / "src").exists() and NOTEBOOK_ROOT.parent != NOTEBOOK_ROOT:
    NOTEBOOK_ROOT = NOTEBOOK_ROOT.parent
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))

from scripts.reporting import generate_experiments_report_notebook as report

ROOT = report.ROOT
bundle = report.load_dataset_bundle()
dataset_facts = bundle["dataset_facts"]
split_summary = bundle["split_summary"]
target_distribution = bundle["target_distribution"]
rqvae_summary = report.collect_rqvae_runs()
cpt_summary = report.collect_cpt_runs()
sft_metrics, sft_diagnostics = report.collect_sft_runs(bundle)
medium_epoch_path = report.SFT_DIR / "sft_gpt2_medium_plus_plus_plus_monitor" / "logs" / "epoch_metrics.json"
medium_epochs = pd.DataFrame(report.read_json(medium_epoch_path)) if medium_epoch_path.exists() else pd.DataFrame()

def show_png(path):
    display(Image(filename=str(path)))

{
    "root": str(ROOT),
    "rqvae_runs": len(rqvae_summary),
    "cpt_checkpoints": len(cpt_summary),
    "sft_metric_rows": len(sft_metrics),
}""",
            outputs=[
                {
                    "output_type": "execute_result",
                    "execution_count": None,
                    "metadata": {},
                    "data": {
                        "text/plain": str(
                            {
                                "root": str(ROOT),
                                "rqvae_runs": len(rqvae_df),
                                "cpt_checkpoints": len(cpt_df),
                                "sft_metric_rows": len(sft_metrics),
                            }
                        )
                    },
                }
            ],
            execution_count=1,
        )
    )

    cells.append(markdown_cell("## 1. Данные, splits и SID-качество"))
    cells.append(
        code_cell(
            """# Train/val/test split summary.
split_summary""",
            outputs=[table_output(bundle["split_summary"])],
            execution_count=2,
        )
    )
    cells.append(
        markdown_cell(
            f"""**Комментарий.** Сплит хронологический leave-last-two на пользователя: `train` содержит почти все взаимодействия, `val` - предпоследний item, `test` - последний item. Это корректный базовый протокол для next-item recommendation. Медианная train-history длина: `{facts["history_len_median"]:.0f}` events; 90-й процентиль: `{facts["history_len_p90"]:.0f}` events. При текущем SFT `MAX_LENGTH=512` в prompt реально помещается существенно меньше 100 событий.
"""
        )
    )
    cells.append(
        code_cell(
            """# Факты по SID и истории пользователей.
dataset_facts""",
            outputs=[text_output(json.dumps(facts, ensure_ascii=False, indent=2))],
            execution_count=3,
        )
    )
    cells.append(
        code_cell(
            """# Распределение размеров splits и длины пользовательской истории.
show_png(report.plot_dataset(bundle))""",
            outputs=[image_output(figures["dataset"])],
            execution_count=4,
        )
    )
    cells.append(
        code_cell(
            """# Насколько target distribution сконцентрирован по популярным items.
target_distribution""",
            outputs=[table_output(bundle["target_distribution"])],
            execution_count=5,
        )
    )

    cells.append(markdown_cell("## 2. RQ-VAE / Semantic ID эксперименты"))
    cells.append(
        markdown_cell(
            f"""Здесь собраны все `runs/*/metrics.json`, где были метрики RQ-VAE. Для PLUM-like пайплайна самый важный критерий на этой стадии - не только reconstruction loss, но и **уникальность SID tuple**, потому что генеративный recommender должен маппить сгенерированный SID обратно в конкретный item.

Лучший экспортированный результат проекта: `SIDs_V1.npy`, uniqueness **{pct(facts["sid_uniqueness"])}**. По локальным run logs лучший RQ-VAE-кандидат: `{best_rqvae.get("run", "n/a")}` с max logged uniqueness `{safe_round(best_rqvae.get("best_sid_uniqueness_pct"), 4)}%`.
"""
        )
    )
    cells.append(
        code_cell(
            """# RQ-VAE runs sorted by best observed full-SID uniqueness.
rqvae_summary""",
            outputs=[table_output(rqvae_view, max_rows=25)],
            execution_count=6,
        )
    )
    cells.append(
        code_cell(
            """# Сравнение лучших RQ-VAE runs по SID uniqueness.
show_png(report.plot_rqvae(rqvae_summary, exported_sid_uniqueness=dataset_facts["sid_uniqueness"]))""",
            outputs=[image_output(figures["rqvae"])],
            execution_count=7,
        )
    )
    cells.append(
        markdown_cell(
            """**Комментарий.** RQ-VAE stage можно считать наиболее зрелой частью проекта: есть progression от слабых 3-level/старых runs к 5-level SID с высокой уникальностью. При этом это не полная PLUM SID-v2 реализация: MovieLens не имеет production-видео модальностей, а признаки являются суррогатом из title/year/genres."""
        )
    )

    cells.append(markdown_cell("## 3. CPT эксперименты GPT-2 / GPT-2 Medium"))
    cells.append(
        markdown_cell(
            """CPT-чекпоинты извлечены из `trainer_state.json`. Эта стадия проверяет, что LLM адаптировалась к SID/user/history/metadata токенам. Важно: **CPT loss нельзя интерпретировать как recommendation metric**. Это language-model objective на смеси behavior и metadata corpus."""
        )
    )
    cells.append(
        code_cell(
            """# CPT checkpoints discovered from trainer_state.json.
cpt_summary""",
            outputs=[table_output(cpt_view, max_rows=30)],
            execution_count=8,
        )
    )
    cells.append(
        code_cell(
            """# Best CPT eval losses by checkpoint.
show_png(report.plot_cpt(cpt_summary))""",
            outputs=[image_output(figures["cpt"])],
            execution_count=9,
        )
    )
    cells.append(
        markdown_cell(
            """**Комментарий.** По CPT видно, что серия GPT-2 Medium дошла до низких eval losses, но SFT результаты ниже показывают: хороший CPT loss не гарантирует хороший next-item ranking. Для отчета это важный разделитель: *domain adaptation happened*, но *recommendation quality still not solved*."""
        )
    )

    cells.append(markdown_cell("## 4. SFT и baseline-контроль"))
    cells.append(
        markdown_cell(
            """SFT должен отвечать на главный вопрос: генерирует ли модель SID следующего item лучше простых baseline. Здесь сравниваются:

- `Popularity`: рекомендуем самые популярные unseen items.
- `sft_gpt2_s_weak_cpt_3epochs`: старый full val/test запуск GPT-2 base.
- `sft_gpt2_medium_plus_plus_plus_monitor`: свежий GPT-2 Medium SFT, пока только epoch monitor на 1024 val examples.

Критическая проверка: Medium SFT на том же val-probe проигрывает popularity baseline по Recall@10 и Recall@20."""
        )
    )
    cells.append(
        code_cell(
            """# SFT and popularity metrics.
sft_metrics""",
            outputs=[table_output(metric_view, max_rows=40)],
            execution_count=10,
        )
    )
    cells.append(
        code_cell(
            """# SFT vs popularity baseline.
show_png(report.plot_sft_metrics(sft_metrics))""",
            outputs=[image_output(figures["sft_metrics"])],
            execution_count=11,
        )
    )
    cells.append(
        markdown_cell(
            f"""**Комментарий.** На val-probe:

- Popularity `Recall@10 = {safe_round(bundle["popularity"]["popularity_val_probe_1024"]["recall@10"], 4)}`;
- Medium SFT epoch-03 `Recall@10 = {safe_round(medium_best_row.get("recall@10"), 4)}`;
- Popularity `Recall@20 = {safe_round(bundle["popularity"]["popularity_val_probe_1024"]["recall@20"], 4)}`;
- Medium SFT epoch-03 `Recall@20 = {safe_round(medium_best_row.get("recall@20"), 4)}`.

Это объективно слабый результат. Дополнительно `Recall@20` у Medium monitor занижен/некорректен из-за `RECSYS_BEAMS=10` при `TOP_K=20`; но даже честный `Recall@10` уже ниже baseline.
"""
        )
    )

    cells.append(markdown_cell("## 5. Medium SFT epoch monitor"))
    cells.append(
        code_cell(
            """# Epoch-level monitor for GPT-2 Medium SFT.
medium_epochs""",
            outputs=[table_output(medium_epochs, max_rows=10)],
            execution_count=12,
        )
    )
    cells.append(
        code_cell(
            """# Loss decreases, ranking barely moves.
show_png(report.plot_sft_epochs(medium_epochs))""",
            outputs=[image_output(figures["sft_epochs"])],
            execution_count=13,
        )
    )
    cells.append(
        code_cell(
            """# Generation diagnostics: candidate list length, coverage, concentration.
sft_diagnostics""",
            outputs=[table_output(diag_view, max_rows=20)],
            execution_count=14,
        )
    )
    cells.append(
        code_cell(
            """# Medium SFT diagnostics across epochs.
show_png(report.plot_sft_diagnostics(sft_diagnostics))""",
            outputs=[image_output(figures["sft_diag"])],
            execution_count=15,
        )
    )
    cells.append(
        markdown_cell(
            """**Комментарий.** Teacher-forcing loss падает, но ranking не улучшается убедительно. Это классический симптом несоответствия LM-objective и retrieval-objective: модель лучше предсказывает target SID tokens в teacher-forcing режиме, но beam search top-K остается недостаточно персонализированным и проигрывает популярности."""
        )
    )

    cells.append(markdown_cell("## 6. Каноничность относительно PLUM и решение по серии"))
    cells.append(
        markdown_cell(
            """**Что сделано по духу PLUM:**

- item -> Semantic ID tokenization;
- LLM tokenizer расширен SID/user/schema tokens;
- CPT на user behavior + item metadata;
- SFT как `p(SID_next | user/history prompt)`;
- trie-constrained beam decoding, invalid SID rate `0.0`;
- ranking evaluation через Recall/NDCG/MRR.

**Что не канонично / слабее оригинального PLUM:**

- нет industrial multi-modal video representation;
- SID-v2 реализован как MovieLens surrogate, не production multimodal SID;
- SFT labels не семплированы по engagement/satisfaction reward;
- `MAX_LENGTH=512`, тогда как в PLUM было около 1536 tokens и примерно 100 watches;
- текущая SFT-серия использует только `5` train targets/user вместо полноценного supervised sliding-window корпуса;
- нет сильных sequential baselines в основном leaderboard;
- Medium monitor считал `Recall@20` при `RECSYS_BEAMS=10`.

**Решение.** Текущую SFT-серию правильно закрыть как negative result. Она полезна как проверка end-to-end pipeline, но не как направление для продолжения без изменения постановки."""
        )
    )
    cells.append(markdown_cell("## 7. Рекомендуемый дальнейший план"))
    cells.append(
        markdown_cell(
            """1. Зафиксировать этот notebook как отчет по текущему состоянию и negative result по SFT.
2. Добавить leaderboard-бейзлайны: Popularity, ItemKNN/co-occurrence, GRU4Rec/SASRec.
3. Если возвращаться к PLUM-style SFT, начинать новую серию:
   - `MAX_LENGTH=1024`;
   - `RECSYS_BEAMS >= 20`, лучше `50` для final eval;
   - full sliding-window train examples, а не `5` targets/user;
   - отдельно логировать `Recall@10`, `NDCG@10`, coverage, novelty/popularity bias;
   - сравнивать каждую эпоху с popularity и ItemKNN.
4. Не тратить GPU на дополнительные эпохи текущего Medium SFT без этих изменений."""
        )
    )

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "name": "python",
                "version": "3.13.1",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset_bundle()
    rqvae_df = collect_rqvae_runs()
    cpt_df = collect_cpt_runs()
    sft_metrics, sft_diag = collect_sft_runs(bundle)

    medium_epoch_path = SFT_DIR / "sft_gpt2_medium_plus_plus_plus_monitor" / "logs" / "epoch_metrics.json"
    medium_epochs = pd.DataFrame(read_json(medium_epoch_path)) if medium_epoch_path.exists() else pd.DataFrame()

    figures = {
        "dataset": plot_dataset(bundle),
        "rqvae": plot_rqvae(rqvae_df, exported_sid_uniqueness=bundle["dataset_facts"]["sid_uniqueness"]),
        "cpt": plot_cpt(cpt_df),
        "sft_metrics": plot_sft_metrics(sft_metrics),
        "sft_epochs": plot_sft_epochs(medium_epochs),
        "sft_diag": plot_sft_diagnostics(sft_diag),
    }

    create_notebook(bundle, rqvae_df, cpt_df, sft_metrics, sft_diag, figures)

    summary = {
        "notebook": str(NOTEBOOK_PATH.relative_to(ROOT)),
        "figures_dir": str(FIG_DIR.relative_to(ROOT)),
        "dataset_facts": bundle["dataset_facts"],
        "rqvae_runs": int(len(rqvae_df)),
        "cpt_checkpoints": int(len(cpt_df)),
        "sft_metric_rows": int(len(sft_metrics)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
