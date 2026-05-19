"""Plot compact Qwen3 PEFT/full-FT and SID depth ablations.

The script reads existing experiment JSON outputs and writes README-ready
figures into docs/assets. It does not recompute metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = ROOT / "docs" / "assets"

BG = "#0b1017"
AX = "#0f141c"
GRID = "#303842"
TEXT = "#edf3fb"
MUTED = "#aab4c0"
GREEN = "#55d465"
BLUE = "#5aa7ff"
PURPLE = "#c99cff"
VIOLET = "#9d6cf2"
GRAY = "#9aa4ad"


def read_json(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def metric_series(path: str):
    rows = read_json(path)
    return rows if isinstance(rows, list) else rows.get("metrics", [])


def beam_metrics(path: str):
    data = read_json(path)
    if isinstance(data, list):
        for row in data:
            if row.get("beam_size") == 20 and row.get("num_return_sequences") == 20:
                return row
        return data[0]
    if "results" in data:
        for row in data["results"]:
            if row.get("beam_size") == 20 and row.get("num_return_sequences") == 20:
                return row
        return data["results"][0]
    return data


def setup_axes(ax):
    ax.set_facecolor(AX)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=12)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)


def save(fig, name: str):
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET_DIR / name, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_qlora_vs_fullft_dynamics():
    qlora = metric_series(
        "data/processed/artifacts/"
        "sft_qwen3_0_6b_qlora32_sid_v2_next_watch_w16_allseen_pat3_v1/metrics.json"
    )
    fullft = metric_series(
        "data/processed/artifacts/"
        "sft_qwen3_0_6b_fullft_cpt_fullft_sid_v2_next_watch_w16_allseen_pat3_v1/metrics.json"
    )

    fig, ax = plt.subplots(figsize=(12, 5.2), facecolor=BG)
    setup_axes(ax)
    ax.set_title("Qwen3-0.6B Validation Dynamics", color=TEXT, fontsize=26, weight="bold", pad=18)

    for rows, label, color, linestyle in [
        (qlora, "QLoRA32 Recall@10", GREEN, "-"),
        (fullft, "Full-FT Recall@10", PURPLE, "-"),
        (qlora, "QLoRA32 NDCG@10", BLUE, "--"),
        (fullft, "Full-FT NDCG@10", GRAY, "--"),
    ]:
        ax.plot(
            [r["epoch"] for r in rows],
            [r["eval_recall@10"] if "Recall" in label else r["eval_ndcg@10"] for r in rows],
            marker="o",
            linewidth=2.6,
            markersize=5.5,
            color=color,
            linestyle=linestyle,
            label=label,
        )

    ax.set_xlabel("Epoch", color=MUTED, fontsize=12)
    ax.set_ylabel("Metric", color=MUTED, fontsize=12)
    ax.set_ylim(0.05, 0.31)
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor=AX,
        edgecolor=GRID,
        labelcolor=TEXT,
        fontsize=11,
    )
    save(fig, "qwen3_0_6b_qlora32_vs_fullft_validation_dynamics.png")


def plot_qlora_vs_fullft_test():
    rows = [
        (
            "Qwen3-0.6B QLoRA32",
            beam_metrics(
                "data/processed/artifacts/"
                "sft_qwen3_0_6b_qlora32_sid_v2_trainval_test_w16_e1_allseen_v1/"
                "test_full_beam20_return20_all_seen_20260519_001408/beam_sweep_results.json"
            ),
            GREEN,
        ),
        (
            "Qwen3-0.6B Full-FT",
            read_json(
                "data/processed/artifacts/"
                "sft_qwen3_0_6b_fullft_cpt_fullft_sid_v2_trainval_test_w16_e7_allseen_v1/"
                "final_test_metrics.json"
            ),
            PURPLE,
        ),
    ]
    metrics = ["recall@1", "recall@5", "recall@10", "ndcg@10", "mrr@10"]
    labels = ["Recall@1", "Recall@5", "Recall@10", "NDCG@10", "MRR@10"]

    fig, ax = plt.subplots(figsize=(12, 5.2), facecolor=BG)
    setup_axes(ax)
    ax.set_title("Qwen3-0.6B Test Metrics", color=TEXT, fontsize=26, weight="bold", pad=18)

    x = range(len(metrics))
    width = 0.34
    for offset, (name, data, color) in zip([-width / 2, width / 2], rows, strict=False):
        values = [data[m] for m in metrics]
        ax.bar([i + offset for i in x], values, width=width, color=color, edgecolor="#1d2731", label=name)
        for i, value in enumerate(values):
            ax.text(i + offset, value + 0.006, f"{value:.3f}", ha="center", va="bottom", color=TEXT, fontsize=10)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, color=TEXT, fontsize=12)
    ax.set_ylim(0, 0.29)
    ax.legend(frameon=True, facecolor=AX, edgecolor=GRID, labelcolor=TEXT, fontsize=11)
    save(fig, "qwen3_0_6b_qlora32_vs_fullft_test_metrics.png")


def plot_codebook_ablation():
    rows = [
        (
            "0.6B / 4 levels",
            beam_metrics(
                "data/processed/artifacts/"
                "sft_qwen3_0_6b_qlora32_sid_v2_trainval_test_w16_e1_allseen_v1/"
                "test_full_beam20_return20_all_seen_20260519_001408/beam_sweep_results.json"
            ),
            VIOLET,
        ),
        (
            "0.6B / 3 levels",
            beam_metrics(
                "data/processed/artifacts/"
                "sft_qwen3_0_6b_qlora32_sid_v2_3codebooks_trainval_test_w16_e3_allseen_v1/"
                "test_full_beam20_return20_all_seen_20260519_021421/beam_sweep_results.json"
            ),
            PURPLE,
        ),
        (
            "4B / 4 levels",
            beam_metrics(
                "data/processed/artifacts/"
                "sft_qwen3_4b_qlora32_sid_v2_trainval_test_w16_e3_allseen_v1/"
                "test_full_beam20_return20_all_seen_20260515_170416/beam_sweep_results.json"
            ),
            BLUE,
        ),
        (
            "4B / 3 levels",
            beam_metrics(
                "data/processed/artifacts/"
                "sft_qwen3_4b_qlora32_sid_v2_3codebooks_trainval_test_w16_e1_allseen_v1/"
                "test_beam20_allseen/beam_sweep_results.json"
            ),
            GREEN,
        ),
    ]

    fig, ax = plt.subplots(figsize=(12, 5.2), facecolor=BG)
    setup_axes(ax)
    ax.set_title("SID Depth Ablation: Held-out Test Recall@10", color=TEXT, fontsize=26, weight="bold", pad=18)

    labels = [r[0] for r in rows]
    values = [r[1]["recall@10"] for r in rows]
    colors = [r[2] for r in rows]
    ax.barh(labels, values, color=colors, edgecolor="#1d2731", height=0.56)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.285)
    ax.set_xlabel("Recall@10", color=MUTED, fontsize=12)
    for y, value in enumerate(values):
        ax.text(value + 0.005, y, f"{value:.4f}", va="center", color=TEXT, fontsize=12)
    save(fig, "qwen3_sid_depth_ablation_test_recall10.png")


def plot_heldout_recall10():
    rows = [
        ("Popularity", 0.0363, "#626c76"),
        ("Content KNN", 0.0675, "#98a2aa"),
        ("ItemKNN", 0.1907, BLUE),
        (
            "Qwen3-0.6B Full-FT",
            read_json(
                "data/processed/artifacts/"
                "sft_qwen3_0_6b_fullft_cpt_fullft_sid_v2_trainval_test_w16_e7_allseen_v1/"
                "final_test_metrics.json"
            )["recall@10"],
            "#b58af0",
        ),
        (
            "Qwen3-0.6B QLoRA32",
            beam_metrics(
                "data/processed/artifacts/"
                "sft_qwen3_0_6b_qlora32_sid_v2_trainval_test_w16_e1_allseen_v1/"
                "test_full_beam20_return20_all_seen_20260519_001408/beam_sweep_results.json"
            )["recall@10"],
            VIOLET,
        ),
        (
            "Qwen3-4B LoRA16",
            beam_metrics(
                "data/processed/artifacts/"
                "sft_qwen3_4b_sid_v2_trainval_test_w16_bestepoch_v1/"
                "test_full_beam20_return20_strict_full_seen_b5/beam_sweep_results.json"
            )["recall@10"],
            "#9d6cf2",
        ),
        (
            "Qwen3-4B QLoRA32",
            beam_metrics(
                "data/processed/artifacts/"
                "sft_qwen3_4b_qlora32_sid_v2_trainval_test_w16_e3_allseen_v1/"
                "test_full_beam20_return20_all_seen_20260515_170416/beam_sweep_results.json"
            )["recall@10"],
            PURPLE,
        ),
        ("BERT4Rec", 0.3066, "#3fb856"),
        ("SASRec", 0.3104, GREEN),
    ]
    rows = sorted(rows, key=lambda row: row[1], reverse=True)

    fig, ax = plt.subplots(figsize=(13, 6.4), facecolor=BG)
    setup_axes(ax)
    ax.set_title("Held-out Test Recall@10", color=TEXT, fontsize=26, weight="bold", pad=18)

    labels = [row[0] for row in rows]
    values = [row[1] for row in rows]
    colors = [row[2] for row in rows]
    ax.barh(labels, values, color=colors, edgecolor="#1d2731", height=0.58)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.34)
    ax.set_xlabel("Recall@10", color=MUTED, fontsize=12)
    for y, value in enumerate(values):
        ax.text(value + 0.006, y, f"{value:.4f}", va="center", color=TEXT, fontsize=12)
    save(fig, "heldout_test_recall10_qwen3_06b_comparison.png")


def main():
    plot_qlora_vs_fullft_dynamics()
    plot_qlora_vs_fullft_test()
    plot_codebook_ablation()
    plot_heldout_recall10()


if __name__ == "__main__":
    main()
