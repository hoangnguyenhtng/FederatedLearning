"""
Evaluation Script for Federated Multi-Modal Recommendation (Amazon Reviews 2023)

This version is aligned with:
- `src/training/federated_training_pipeline.py` (model creation + checkpoint format)
- `src/data_generation/amazon_dataloader.py` (Amazon per-client data)

It evaluates the *global* checkpoint (shared + personal head as saved) on each client test split.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from src.data_generation.amazon_dataloader import get_amazon_dataloaders
from src.models.multimodal_encoder import MultiModalEncoder
from src.models.recommendation_model import FedPerRecommender
from src.training.training_utils import calculate_metrics


@dataclass
class EvalPaths:
    experiment_dir: Path
    evaluation_dir: Path
    checkpoint_path: Path


def _infer_experiment_dir(config: dict) -> Path:
    exp_name = config.get("experiment", {}).get("name", "fedper_multimodal_v1")
    exp_root = Path(config.get("paths", {}).get("experiments_dir", "experiments"))
    return exp_root / exp_name


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(str(path), map_location=device)


def _create_model_from_config(config: dict, device: torch.device) -> FedPerRecommender:
    model_config = config["model"]

    encoder = MultiModalEncoder(
        text_dim=model_config.get("text_embedding_dim", 384),
        image_dim=model_config.get("image_embedding_dim", 2048),
        behavior_dim=model_config.get("behavior_embedding_dim", 32),
        hidden_dim=model_config.get("hidden_dim", 256),
        output_dim=384,
    )

    model = FedPerRecommender(
        multimodal_encoder=encoder,
        shared_hidden_dims=model_config.get("shared_hidden_dims", [512, 256, 128]),
        personal_hidden_dims=model_config.get("personal_hidden_dims", [64, 32]),
        num_classes=model_config.get("num_classes", 5),
        dropout=model_config.get("dropout", 0.2),
    )

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _evaluate_loader(
    model: nn.Module,
    loader,
    device: torch.device,
    compute_all_metrics: bool = True,
) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    for batch in loader:
        # Amazon format keys
        if "text_embedding" in batch:
            text_emb = batch["text_embedding"].to(device)
        else:
            text_emb = torch.randn(loader.batch_size or len(next(iter(loader))["rating"]), 384, device=device)

        if "image_embedding" in batch:
            image_emb = batch["image_embedding"].to(device)
        elif "image_features" in batch:
            image_emb = batch["image_features"].to(device)
        else:
            image_emb = torch.randn(text_emb.shape[0], 2048, device=device)

        behavior_feat = batch["behavior_features"].to(device)

        if "label" in batch:
            targets = batch["label"].to(device)
        else:
            targets = (batch["rating"].to(device) - 1).clamp(0, 4)

        logits = model(text_emb, image_emb, behavior_feat)
        loss = criterion(logits, targets)

        total_loss += float(loss.item())
        all_logits.append(logits.detach())
        all_targets.append(targets.detach())

    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)

    metrics = {"loss": total_loss / max(1, len(loader))}
    metrics.update(calculate_metrics(logits_cat, targets_cat, compute_all=compute_all_metrics))
    return metrics


def evaluate_checkpoint_on_amazon_clients(
    config_path: Path,
    experiment_dir: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    amazon_dir: Path = Path("data/amazon_2023_processed"),
    save: bool = True,
) -> Tuple[pd.DataFrame, dict, EvalPaths]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = experiment_dir or _infer_experiment_dir(config)
    eval_dir = exp_dir / "evaluation"
    ckpt_path = checkpoint_path or (exp_dir / "models" / "global_model_final.pt")

    paths = EvalPaths(experiment_dir=exp_dir, evaluation_dir=eval_dir, checkpoint_path=ckpt_path)

    # Load model + checkpoint
    model = _create_model_from_config(config, device=device)
    ckpt = _load_checkpoint(paths.checkpoint_path, device=device)
    if "model_state_dict" not in ckpt:
        raise ValueError(f"Invalid checkpoint format (missing model_state_dict): {paths.checkpoint_path}")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # Load Amazon per-client dataloaders
    num_clients = int(config["federated"]["num_clients"])
    batch_size = int(config["training"]["batch_size"])
    test_split = float(config["training"].get("test_split", 0.2))

    dataloaders = get_amazon_dataloaders(
        num_clients=num_clients,
        data_dir=str(amazon_dir),
        batch_size=batch_size,
        test_split=test_split,
    )

    results: List[Dict[str, float]] = []
    for client_id in sorted(dataloaders.keys()):
        _, test_loader = dataloaders[client_id]
        metrics = _evaluate_loader(model=model, loader=test_loader, device=device, compute_all_metrics=True)
        results.append({"client_id": client_id, **metrics, "num_test_samples": len(test_loader.dataset)})
        print(
            f"Client {client_id:02d}: "
            f"loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f} "
            f"ndcg@10={metrics.get('ndcg@10', 0.0):.4f} mrr={metrics.get('mrr', 0.0):.4f}"
        )

    df = pd.DataFrame(results)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(paths.checkpoint_path),
        "device": str(device),
        "num_clients_evaluated": int(df.shape[0]),
        "overall": {
            "mean_loss": float(df["loss"].mean()),
            "mean_accuracy": float(df["accuracy"].mean()),
            "std_accuracy": float(df["accuracy"].std(ddof=0)) if df.shape[0] > 1 else 0.0,
            "mean_precision": float(df.get("precision", pd.Series([np.nan])).mean()),
            "mean_recall": float(df.get("recall", pd.Series([np.nan])).mean()),
            "mean_ndcg@10": float(df.get("ndcg@10", pd.Series([np.nan])).mean()),
            "mean_mrr": float(df.get("mrr", pd.Series([np.nan])).mean()),
        },
    }

    if save:
        paths.evaluation_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(paths.evaluation_dir / "amazon_client_results.csv", index=False)
        with open(paths.evaluation_dir / "amazon_evaluation_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\n📝 Saved: {paths.evaluation_dir / 'amazon_client_results.csv'}")
        print(f"📝 Saved: {paths.evaluation_dir / 'amazon_evaluation_summary.json'}")

    return df, summary, paths


def main():
    parser = argparse.ArgumentParser(description="Evaluate FedPer global checkpoint on Amazon clients")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--amazon_dir", type=str, default="data/amazon_2023_processed")
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--no_save", action="store_true")

    args = parser.parse_args()

    df, summary, _ = evaluate_checkpoint_on_amazon_clients(
        config_path=Path(args.config),
        experiment_dir=Path(args.experiment_dir) if args.experiment_dir else None,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        amazon_dir=Path(args.amazon_dir),
        save=not args.no_save,
    )

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY (AMAZON)")
    print("=" * 70)
    overall = summary["overall"]
    print(f"Clients evaluated: {summary['num_clients_evaluated']}")
    print(f"Mean loss: {overall['mean_loss']:.4f}")
    print(f"Mean accuracy: {overall['mean_accuracy']:.4f} (std={overall['std_accuracy']:.4f})")
    print(f"Mean NDCG@10: {overall['mean_ndcg@10']:.4f}")
    print(f"Mean MRR: {overall['mean_mrr']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()