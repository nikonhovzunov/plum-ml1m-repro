from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def find_root(start: Path) -> Path:
    root = start.resolve()
    while not (root / "src").exists() and root.parent != root:
        root = root.parent
    return root


ROOT = find_root(Path.cwd())
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

evaluate_rankings = importlib.import_module("plum_ml1m.metrics").evaluate_rankings

TRAIN_PATH = ROOT / "data/processed/splits/train.parquet"
VAL_PATH = ROOT / "data/processed/splits/val.parquet"
TEST_PATH = ROOT / "data/processed/splits/test.parquet"
EMBEDDING_PATH = ROOT / "data/processed/item_features/qwen4b_audited_v1_meta_desc_embeddings.npz"
DEFAULT_OUTPUT = ROOT / "reports/sasrec_multimodal/sasrec_qwen_concat_w16"
K_VALUES = (1, 5, 10, 20, 50, 100)


def read_split(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path).sort_values(
        ["user_idx", "timestamp", "pos", "item_idx"], kind="mergesort"
    )


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def load_content(path: Path) -> np.ndarray:
    bundle = np.load(path)
    item_idx = bundle["item_idx"].astype(np.int64)
    meta = bundle["meta"].astype(np.float32)
    desc = bundle["description"].astype(np.float32)
    if not np.array_equal(item_idx, np.arange(len(item_idx))):
        raise ValueError("Embedding item_idx must be contiguous and sorted from 0")
    return l2(np.concatenate([l2(meta), l2(desc)], axis=1)).astype(np.float32)


def train_sequences_from_context(
    df: pd.DataFrame,
    window: int,
    min_history_len: int,
) -> list[list[int]]:
    rows = []
    seq_len = window + 1
    for _, group in df.groupby("user_id", sort=False):
        group = group.sort_values(["timestamp", "pos", "item_idx"], kind="mergesort")
        items = group["item_idx"].astype(int).tolist()
        if len(items) <= min_history_len:
            continue
        for end in range(window, len(items)):
            seq = items[end - window : end + 1]
            if len(seq) == seq_len:
                rows.append(seq)
    return rows


def eval_examples_from_context(
    context_df: pd.DataFrame,
    target_df: pd.DataFrame,
    window: int,
    min_history_len: int,
) -> list[dict[str, Any]]:
    context_by_user = {
        int(user_id): group.sort_values(
            ["timestamp", "pos", "item_idx"], kind="mergesort"
        )
        for user_id, group in context_df.groupby("user_id", sort=False)
    }
    rows = []
    target_df = target_df.sort_values(["user_idx", "timestamp", "pos", "item_idx"], kind="mergesort")
    for row in target_df.itertuples(index=False):
        hist = context_by_user.get(int(row.user_id))
        if hist is None:
            continue
        items = hist["item_idx"].astype(int).tolist()
        if len(items) < min_history_len:
            continue
        rows.append(
            {
                "user_id": int(row.user_id),
                "history": items[-window:],
                "seen": items,
                "target": int(row.item_idx),
            }
        )
    return rows


class SASRecTrainDataset(Dataset):
    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.rows[idx]
        return (
            torch.tensor(seq[:-1], dtype=torch.long),
            torch.tensor(seq[1:], dtype=torch.long),
        )


class MultimodalSASRec(nn.Module):
    def __init__(
        self,
        content_matrix: np.ndarray,
        seq_len: int,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_items = content_matrix.shape[0]
        self.register_buffer("content_matrix", torch.tensor(content_matrix, dtype=torch.float32))
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1),
        )
        self.item_emb = nn.Embedding(self.n_items, hidden_dim)
        self.pos_emb = nn.Embedding(seq_len, hidden_dim)
        self.content_proj = nn.Linear(content_matrix.shape[1], hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, self.n_items)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        content = self.content_proj(self.content_matrix[input_ids])
        x = self.item_emb(input_ids) + content + self.pos_emb(pos)
        x = self.encoder(x, mask=self.causal_mask[: input_ids.size(1), : input_ids.size(1)])
        x = self.norm(x)
        return self.out(x)

    def score_next(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self(input_ids)[:, -1, :]


def evaluate(
    model: MultimodalSASRec,
    rows: list[dict[str, Any]],
    batch_size: int,
    device: str,
    name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    records = []
    max_k = max(K_VALUES)
    with torch.inference_mode():
        for start in tqdm(range(0, len(rows), batch_size), desc=name):
            batch = rows[start : start + batch_size]
            ids = torch.tensor([row["history"] for row in batch], dtype=torch.long, device=device)
            logits = model.score_next(ids)
            for i, row in enumerate(batch):
                scores = logits[i].detach().clone()
                seen = torch.tensor(row["seen"], dtype=torch.long, device=device)
                scores[seen] = -torch.inf
                candidates = torch.topk(scores, k=min(max_k, scores.numel())).indices.cpu().tolist()
                records.append(
                    {
                        "user_id": int(row["user_id"]),
                        "target_item_idx": int(row["target"]),
                        "candidates": candidates,
                    }
                )
    metrics = evaluate_rankings(records, K_VALUES)
    metrics["n"] = len(records)
    return metrics, records


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True


def cuda_stats() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "cuda_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "cuda_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }


def train_run(
    args: argparse.Namespace,
    context: pd.DataFrame,
    target: pd.DataFrame,
    content: np.ndarray,
    split: str,
    epochs: int,
    early_stop: bool,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    train_rows = train_sequences_from_context(context, args.window, args.min_history_len)
    eval_rows = eval_examples_from_context(context, target, args.window, args.min_history_len)
    train_loader = DataLoader(
        SASRecTrainDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device == "cuda",
    )
    model = MultimodalSASRec(
        content,
        seq_len=args.window,
        hidden_dim=args.hidden_dim,
        n_layers=args.layers,
        n_heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_metric = -1.0
    best_epoch = 0
    bad_epochs = 0
    best_path = output_dir / f"best_model_{split}.pt"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "model": "multimodal_sasrec",
        "split": split,
        "context_source": "train" if split == "val" else "train+val",
        "target_source": split,
        "window": args.window,
        "loss": "causal_next_item_all_positions",
        "seen_filter": "all_prior_history",
        "content": "l2(concat(l2(meta), l2(description)))",
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "heads": args.heads,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "epochs": epochs,
        "early_stop": early_stop,
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "content_dim": int(content.shape[1]),
        "parameters": int(sum(p.numel() for p in model.parameters())),
    }
    (output_dir / f"run_config_{split}.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_config, ensure_ascii=False, indent=2))

    for epoch in range(1, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model.train()
        losses = []
        started = time.time()
        progress = tqdm(train_loader, desc=f"{split} epoch {epoch}/{epochs}")
        for input_ids, targets in progress:
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(input_ids)
                loss = criterion(logits.reshape(-1, model.n_items), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            value = float(loss.detach().cpu())
            losses.append(value)
            progress.set_postfix({"loss": f"{value:.4f}"})

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_seconds": time.time() - started,
            **cuda_stats(),
        }
        if early_stop:
            metrics, _ = evaluate(model, eval_rows, args.eval_batch_size, device, name=f"eval {split}")
            row.update(metrics)
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))
        pd.DataFrame(history).to_csv(output_dir / f"history_{split}.csv", index=False)

        if row.get("cuda_reserved_gb", 0.0) > args.max_reserved_gb:
            raise RuntimeError(
                f"CUDA reserved memory exceeded limit: {row['cuda_reserved_gb']:.2f} GB"
            )

        if early_stop:
            if row["recall@10"] > best_metric:
                best_metric = row["recall@10"]
                best_epoch = epoch
                bad_epochs = 0
                torch.save(model.state_dict(), best_path)
            else:
                bad_epochs += 1
            if bad_epochs >= args.patience:
                break
        else:
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    final_metrics, records = evaluate(model, eval_rows, args.eval_batch_size, device, name=f"final {split}")
    if not early_stop:
        best_metric = final_metrics["recall@10"]
    report = {
        "split": split,
        "best_epoch": int(best_epoch),
        "best_recall@10": float(best_metric),
        "metrics": final_metrics,
        "history_window": args.window,
        "seen_filter": "all_prior_history",
        "context_source": "train" if split == "val" else "train+val",
        "target_source": split,
        **cuda_stats(),
    }
    (output_dir / f"metrics_{split}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "user_id": rec["user_id"],
                "target_item_idx": rec["target_item_idx"],
                **{f"candidate_{i + 1}": item for i, item in enumerate(rec["candidates"])},
            }
            for rec in records
        ]
    ).to_parquet(output_dir / f"predictions_{split}.parquet", index=False)
    return history, report, best_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--min-history-len", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--max-reserved-gb", type=float, default=14.0)
    parser.add_argument("--test-epochs", type=int, default=0)
    parser.add_argument("--only-test", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = read_split(TRAIN_PATH)
    val = read_split(VAL_PATH)
    test = read_split(TEST_PATH)
    content = load_content(EMBEDDING_PATH)

    if args.only_test:
        if args.test_epochs <= 0:
            raise ValueError("--only-test requires --test-epochs")
        test_context = pd.concat([train, val], ignore_index=True)
        test_history, test_report, _ = train_run(
            args=args,
            context=test_context,
            target=test,
            content=content,
            split="test",
            epochs=args.test_epochs,
            early_stop=False,
            output_dir=output_dir,
        )
        summary = {
            "run_name": output_dir.name,
            "test": test_report,
            "test_epochs": len(test_history),
            "protocol": {
                "test": "train+val -> test for validation-selected epoch count",
                "seen_filter": "all_prior_history",
                "history_window": args.window,
            },
        }
        (output_dir / "summary_test_only.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    val_history, val_report, _ = train_run(
        args=args,
        context=train,
        target=val,
        content=content,
        split="val",
        epochs=args.epochs,
        early_stop=True,
        output_dir=output_dir,
    )
    test_epochs = args.test_epochs if args.test_epochs > 0 else int(val_report["best_epoch"])
    test_context = pd.concat([train, val], ignore_index=True)
    test_history, test_report, _ = train_run(
        args=args,
        context=test_context,
        target=test,
        content=content,
        split="test",
        epochs=test_epochs,
        early_stop=False,
        output_dir=output_dir,
    )
    summary = {
        "run_name": output_dir.name,
        "val": val_report,
        "test": test_report,
        "val_epochs": len(val_history),
        "test_epochs": len(test_history),
        "protocol": {
            "val": "train -> val with early stopping",
            "test": "train+val -> test for validation-selected epoch count",
            "seen_filter": "all_prior_history",
            "history_window": args.window,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
