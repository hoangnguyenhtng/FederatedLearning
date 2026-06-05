"""Đánh giá checkpoint federated trên từng client (Amazon hoặc synthetic)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.multimodal_encoder import MultiModalEncoder
from src.models.recommendation_model import FedPerRecommender
from src.training.training_utils import (
    MetricsCalculator,
    calculate_metrics,
    load_checkpoint,
    resolve_amazon_federated_data_dir,
    experiments_base_dir,
)


def _build_model(model_cfg: dict) -> FedPerRecommender:
    """Create FedPerRecommender identical to training pipeline."""
    encoder = MultiModalEncoder(
        text_dim=model_cfg.get("text_embedding_dim", 384),
        image_dim=model_cfg.get("image_embedding_dim", 2048),
        behavior_dim=model_cfg.get("behavior_embedding_dim",
                                    model_cfg.get("behavior_dim", 32)),
        hidden_dim=model_cfg.get("hidden_dim",
                                  model_cfg.get("multimodal_hidden_dim", 256)),
        output_dim=model_cfg.get("multimodal_output_dim", 384),
    )
    return FedPerRecommender(
        multimodal_encoder=encoder,
        shared_hidden_dims=model_cfg.get("shared_hidden_dims", [512, 256, 128]),
        personal_hidden_dims=model_cfg.get("personal_hidden_dims", [64, 32]),
        num_classes=model_cfg.get("num_classes", 5),
        dropout=model_cfg.get("dropout", 0.2),
    )


class FederatedEvaluator:
    """Evaluate federated model performance across clients."""

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        experiment_dir: Optional[str] = None,
    ):
        resolved = Path(config_path)
        if not resolved.is_absolute():
            resolved = (project_root / resolved).resolve()

        with open(resolved, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if experiment_dir:
            self.experiment_dir = Path(experiment_dir)
        else:
            base = experiments_base_dir(self.config, cwd=project_root)
            candidates = sorted(base.glob("fedper_*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                self.experiment_dir = candidates[0]
            else:
                self.experiment_dir = base / "evaluation_standalone"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        print(f"📊 Evaluator initialized")
        print(f"   Config: {resolved}")
        print(f"   Experiment: {self.experiment_dir}")
        print(f"   Device: {self.device}")

    def load_model(self, checkpoint_path: Optional[str] = None) -> nn.Module:
        """Load trained model from checkpoint (auto-detect if path not given)."""
        model = _build_model(self.config.get("model", {}))
        model.to(self.device)

        if checkpoint_path is None:
            ckpt_candidates = sorted(
                self.experiment_dir.glob("**/global_model_final.pt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not ckpt_candidates:
                base = experiments_base_dir(self.config, cwd=project_root)
                ckpt_candidates = sorted(
                    base.glob("**/global_model_final.pt"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            if ckpt_candidates:
                checkpoint_path = str(ckpt_candidates[0])

        if checkpoint_path and Path(checkpoint_path).exists():
            load_checkpoint(model, optimizer=None, path=checkpoint_path, device=self.device)
            print(f"✅ Loaded checkpoint: {checkpoint_path}")
        else:
            print("⚠️  No checkpoint found — evaluating random-initialized model")
            print("   Results will reflect untrained baseline performance")

        model.eval()
        return model

    def _personalize_head(self, model: nn.Module, train_loader, epochs: int = 3) -> None:
        """Adapt personal head to latest shared weights (standard FedPer eval protocol)."""
        if epochs <= 0:
            return

        shared_names = set(model.get_shared_parameters().keys())
        for name, param in model.named_parameters():
            param.requires_grad = name not in shared_names

        model.train()
        optimizer = torch.optim.Adam(
            [p for p in model.personal_head.parameters() if p.requires_grad],
            lr=0.001,
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for i, batch_data in enumerate(train_loader):
                if i >= 10:  # Limit to 10 batches (320 samples) for fast personalization on CPU
                    break
                if isinstance(batch_data, dict):
                    if "image_embedding" in batch_data:
                        image_emb = batch_data["image_embedding"].to(self.device)
                    elif "image_features" in batch_data:
                        image_emb = batch_data["image_features"].to(self.device)
                    else:
                        bs = batch_data.get("label", batch_data.get("rating", torch.tensor([1]))).shape[0]
                        image_emb = torch.randn(bs, 2048, device=self.device)

                    behavior_feat = batch_data["behavior_features"].to(self.device)
                    labels = batch_data.get("label", batch_data["rating"] - 1).to(self.device)
                    labels = torch.clamp(labels, 0, 2)  # 3-class sentiment

                    bs = behavior_feat.shape[0]
                    if behavior_feat.shape[-1] != 32:
                        if behavior_feat.shape[-1] < 32:
                            pad = torch.zeros(bs, 32 - behavior_feat.shape[-1], device=self.device)
                            behavior_feat = torch.cat([behavior_feat, pad], dim=1)
                        else:
                            behavior_feat = behavior_feat[:, :32]

                    if "text_embedding" in batch_data:
                        text_emb = batch_data["text_embedding"].to(self.device)
                    else:
                        text_emb = torch.randn(bs, 384, device=self.device)

                    if image_emb.shape[-1] != 2048:
                        image_emb = torch.randn(bs, 2048, device=self.device)
                else:
                    text_emb = batch_data[0].to(self.device)
                    image_emb = batch_data[1].to(self.device)
                    behavior_feat = batch_data[2].to(self.device)
                    labels = batch_data[3].to(self.device)
                    labels = torch.clamp(labels, 0, 2)  # 3-class sentiment

                optimizer.zero_grad()
                logits = model(text_emb, image_emb, behavior_feat)
                num_classes = logits.shape[1]
                labels_clamped = torch.clamp(labels, 0, num_classes - 1)
                loss = criterion(logits, labels_clamped)
                loss.backward()
                optimizer.step()

        for param in model.parameters():
            param.requires_grad = True

    def _load_dataloaders(self) -> Dict[int, tuple]:
        """Load federated dataloaders (Amazon or synthetic)."""
        num_clients = self.config["federated"]["num_clients"]
        batch_size = self.config["training"]["batch_size"]
        test_split = self.config["training"].get("test_split", 0.2)

        amazon_dir = resolve_amazon_federated_data_dir(self.config, cwd=project_root)

        if amazon_dir is not None:
            print(f"📂 Using Amazon data: {amazon_dir}")
            from src.data_generation.amazon_dataloader import get_amazon_dataloaders

            # Nhiều người train với config 40 client nhưng chạy report với config.yaml (10).
            # Nếu trên đĩa có nhiều client_*/data.pkl hơn num_clients trong YAML → dùng hết cho báo cáo.
            n_on_disk = sum(1 for _ in amazon_dir.glob("client_*/data.pkl"))
            if n_on_disk > num_clients:
                print(
                    f"📊 Config có num_clients={num_clients}, nhưng tìm thấy {n_on_disk} file data.pkl — "
                    f"dùng {n_on_disk} client cho evaluation/report."
                )
                num_clients = n_on_disk

            return get_amazon_dataloaders(
                num_clients=num_clients,
                data_dir=str(amazon_dir),
                batch_size=batch_size,
                test_split=test_split,
            )

        paths_cfg = self.config.get("paths") or {}
        synthetic_dir = (project_root / paths_cfg.get("data_dir", "data") / "simulated_clients").resolve()
        if synthetic_dir.exists():
            print(f"📂 Using synthetic data: {synthetic_dir}")
            from src.data_generation.federated_dataloader import get_federated_dataloaders
            loaders_list = get_federated_dataloaders(
                num_clients=num_clients,
                data_dir=synthetic_dir,
                batch_size=batch_size,
                test_split=test_split,
            )
            dataloaders = {}
            for cid, loaders in enumerate(loaders_list):
                if loaders and len(loaders) == 2 and loaders[0] and loaders[1]:
                    dataloaders[cid] = (loaders[0], loaders[1])
            return dataloaders

        raise FileNotFoundError(
            f"No data found. Run process_amazon_data.py first.\n"
            f"  Checked: {amazon_dir}, {synthetic_dir}"
        )

    def _evaluate_client(
        self,
        model: nn.Module,
        test_loader,
    ) -> Dict[str, float]:
        """Evaluate model on one client's test set."""
        model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_data in test_loader:
                if isinstance(batch_data, dict):
                    if "image_embedding" in batch_data:
                        image_emb = batch_data["image_embedding"].to(self.device)
                    elif "image_features" in batch_data:
                        image_emb = batch_data["image_features"].to(self.device)
                    else:
                        bs = batch_data.get("label", batch_data.get("rating", torch.tensor([1]))).shape[0]
                        image_emb = torch.randn(bs, 2048, device=self.device)

                    behavior_feat = batch_data["behavior_features"].to(self.device)
                    labels = batch_data.get("label", batch_data["rating"] - 1).to(self.device)
                    labels = torch.clamp(labels, 0, 2)  # 3-class sentiment

                    bs = behavior_feat.shape[0]
                    if behavior_feat.shape[-1] != 32:
                        if behavior_feat.shape[-1] < 32:
                            pad = torch.zeros(bs, 32 - behavior_feat.shape[-1], device=self.device)
                            behavior_feat = torch.cat([behavior_feat, pad], dim=1)
                        else:
                            behavior_feat = behavior_feat[:, :32]

                    if "text_embedding" in batch_data:
                        text_emb = batch_data["text_embedding"].to(self.device)
                    else:
                        text_emb = torch.randn(bs, 384, device=self.device)

                    if image_emb.shape[-1] != 2048:
                        image_emb = torch.randn(bs, 2048, device=self.device)
                else:
                    text_emb = batch_data[0].to(self.device)
                    image_emb = batch_data[1].to(self.device)
                    behavior_feat = batch_data[2].to(self.device)
                    labels = batch_data[3].to(self.device)
                    labels = torch.clamp(labels, 0, 2)  # 3-class sentiment

                logits = model(text_emb, image_emb, behavior_feat)
                num_classes = logits.shape[1]
                labels_clamped = torch.clamp(labels, 0, num_classes - 1)

                loss = criterion(logits, labels_clamped)
                total_loss += loss.item()

                _, predicted = torch.max(logits, 1)
                total += labels_clamped.size(0)
                correct += (predicted == labels_clamped).sum().item()

                all_preds.append(logits.cpu())
                all_targets.append(labels_clamped.cpu())

        avg_loss = total_loss / max(len(test_loader), 1)
        accuracy = correct / max(total, 1)

        num_cls = logits.shape[1] if all_preds else 3
        all_preds_t = torch.cat(all_preds) if all_preds else torch.zeros(1, num_cls)
        all_targets_t = torch.cat(all_targets) if all_targets else torch.zeros(1).long()

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "num_samples": total,
        }

        try:
            extra = calculate_metrics(all_preds_t, all_targets_t, compute_all=True)
            metrics.update(extra)
        except Exception:
            pass

        return metrics

    def _load_single_client_dataloader(self, cid: int) -> Optional[tuple]:
        """Load dataloaders for a single client to save memory (lazy-load)."""
        batch_size = self.config["training"]["batch_size"]
        test_split = self.config["training"].get("test_split", 0.2)

        amazon_dir = resolve_amazon_federated_data_dir(self.config, cwd=project_root)
        if amazon_dir is not None:
            client_path = amazon_dir / f"client_{cid}" / "data.pkl"
            if not client_path.exists():
                return None
            
            import pandas as pd
            from src.data_generation.amazon_dataloader import AmazonDataset
            from torch.utils.data import DataLoader
            
            try:
                client_df = pd.read_pickle(client_path)
                n_test = int(len(client_df) * test_split)
                # Shuffle
                client_df = client_df.sample(frac=1, random_state=42 + cid).reset_index(drop=True)
                test_df = client_df[:n_test]
                train_df = client_df[n_test:]
                
                train_ds = AmazonDataset(train_df)
                test_ds = AmazonDataset(test_df)
                
                train_loader = DataLoader(
                    train_ds, batch_size=batch_size, shuffle=True,
                    num_workers=0, pin_memory=False, drop_last=False
                )
                test_loader = DataLoader(
                    test_ds, batch_size=batch_size, shuffle=False,
                    num_workers=0, pin_memory=False, drop_last=False
                )
                return train_loader, test_loader
            except Exception as e:
                print(f"  ❌ Failed to load client {cid}: {e}")
                return None
        else:
            # Synthetic
            paths_cfg = self.config.get("paths") or {}
            synthetic_dir = (project_root / paths_cfg.get("data_dir", "data") / "simulated_clients").resolve()
            if synthetic_dir.exists():
                from src.data_generation.federated_dataloader import get_federated_dataloaders
                loaders_list = get_federated_dataloaders(
                    num_clients=cid + 1,
                    data_dir=synthetic_dir,
                    batch_size=batch_size,
                    test_split=test_split,
                )
                if len(loaders_list) > cid:
                    loaders = loaders_list[cid]
                    if loaders and len(loaders) == 2 and loaders[0] and loaders[1]:
                        return loaders[0], loaders[1]
            return None

    def evaluate_all_clients(
        self,
        model: nn.Module,
        client_ids: Optional[List[int]] = None,
        personalize_epochs: int = 3,
    ) -> List[Dict]:
        """Evaluate model on all (or selected) clients."""
        # Determine client_ids if not specified
        if client_ids is None:
            amazon_dir = resolve_amazon_federated_data_dir(self.config, cwd=project_root)
            if amazon_dir is not None:
                client_ids = [int(p.name.split("_")[1]) for p in amazon_dir.glob("client_*")]
                client_ids = sorted(client_ids)
            else:
                paths_cfg = self.config.get("paths") or {}
                synthetic_dir = (project_root / paths_cfg.get("data_dir", "data") / "simulated_clients").resolve()
                if synthetic_dir.exists():
                    client_ids = [int(p.name.split("_")[1]) for p in synthetic_dir.glob("client_*")]
                    client_ids = sorted(client_ids)
                else:
                    client_ids = list(range(self.config["federated"]["num_clients"]))

        print(f"\n📊 Evaluating {len(client_ids)} clients sequentially (optimized memory)...")
        results = []

        for cid in client_ids:
            loaders = self._load_single_client_dataloader(cid)
            if loaders is None:
                print(f"  ⚠️  Client {cid}: no data, skipping")
                continue

            train_loader, test_loader = loaders

            # Load personalized head for this client if state file exists
            state_path = self.experiment_dir / "client_states" / f"client_{cid}.pt"
            if state_path.exists():
                try:
                    payload = torch.load(state_path, map_location=str(self.device), weights_only=False)
                    personal = payload.get("personal") or {}
                    current = model.state_dict()
                    for name, param in personal.items():
                        if name in current:
                            current[name] = param.to(self.device)
                    model.load_state_dict(current, strict=False)
                    print(f"  Client {cid}: Loaded personalized head from {state_path.name}")
                except Exception as e:
                    print(f"  ⚠️  Client {cid}: Could not load personal state: {e}")
            else:
                print(f"  ⚠️  Client {cid}: No personalized state found, using global head")

            # ✅ Personalize the head before testing (standard FedPer eval protocol)
            if personalize_epochs > 0:
                print(f"  Client {cid}: Personalizing head for {personalize_epochs} epochs...")
                self._personalize_head(model, train_loader, epochs=personalize_epochs)

            metrics = self._evaluate_client(model, test_loader)
            metrics["client_id"] = cid
            results.append(metrics)

            print(
                f"  Client {cid}: "
                f"Acc={metrics['accuracy']:.4f}, "
                f"Loss={metrics['loss']:.4f}, "
                f"Samples={metrics['num_samples']}"
            )

            # ✅ Free memory by garbage collecting loaders
            del train_loader, test_loader, loaders
            import gc
            gc.collect()

        return results

    def generate_report(
        self,
        results: List[Dict],
        save_dir: Optional[Path] = None,
    ) -> Dict:
        """Generate and save evaluation report."""
        if save_dir is None:
            save_dir = self.experiment_dir / "evaluation"
        save_dir.mkdir(parents=True, exist_ok=True)

        if not results:
            print("⚠️  No results to report")
            return {}

        accuracies = [r["accuracy"] for r in results]
        losses = [r["loss"] for r in results]

        report = {
            "timestamp": datetime.now().isoformat(),
            "num_clients_evaluated": len(results),
            "overall_metrics": {
                "mean_accuracy": float(np.mean(accuracies)),
                "std_accuracy": float(np.std(accuracies)),
                "min_accuracy": float(np.min(accuracies)),
                "max_accuracy": float(np.max(accuracies)),
                "mean_loss": float(np.mean(losses)),
            },
            "per_client": results,
        }

        for metric_name in ["precision", "recall", "ndcg@10", "mrr"]:
            vals = [r.get(metric_name) for r in results if r.get(metric_name) is not None]
            if vals:
                report["overall_metrics"][f"mean_{metric_name}"] = float(np.mean(vals))

        report_path = save_dir / "evaluation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n📝 Report saved: {report_path}")

        try:
            import pandas as pd
            df = pd.DataFrame(results)
            csv_path = save_dir / "client_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"📝 CSV saved: {csv_path}")
        except ImportError:
            pass

        return report

    def visualize_results(
        self,
        results: List[Dict],
        save_dir: Optional[Path] = None,
    ):
        """Create evaluation plots."""
        if save_dir is None:
            save_dir = self.experiment_dir / "evaluation"
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            client_ids = [r["client_id"] for r in results]
            accuracies = [r["accuracy"] for r in results]
            losses = [r["loss"] for r in results]

            axes[0].bar(client_ids, accuracies, color="#4ECDC4", edgecolor="#333")
            axes[0].axhline(y=np.mean(accuracies), color="r", linestyle="--",
                            label=f"Mean: {np.mean(accuracies):.3f}")
            axes[0].set_xlabel("Client ID")
            axes[0].set_ylabel("Accuracy")
            axes[0].set_title("Per-Client Accuracy")
            axes[0].legend()
            axes[0].grid(axis="y", alpha=0.3)

            axes[1].bar(client_ids, losses, color="#FF6B6B", edgecolor="#333")
            axes[1].axhline(y=np.mean(losses), color="r", linestyle="--",
                            label=f"Mean: {np.mean(losses):.3f}")
            axes[1].set_xlabel("Client ID")
            axes[1].set_ylabel("Loss")
            axes[1].set_title("Per-Client Loss")
            axes[1].legend()
            axes[1].grid(axis="y", alpha=0.3)

            metrics_names = ["accuracy"]
            metric_vals = [accuracies]
            for m in ["precision", "recall"]:
                vals = [r.get(m) for r in results if r.get(m) is not None]
                if vals:
                    metrics_names.append(m)
                    metric_vals.append(vals)

            axes[2].boxplot(metric_vals, labels=metrics_names)
            axes[2].set_ylabel("Score")
            axes[2].set_title("Metrics Distribution")
            axes[2].grid(axis="y", alpha=0.3)

            plt.tight_layout()
            viz_path = save_dir / "evaluation_results.png"
            plt.savefig(viz_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"📊 Visualization saved: {viz_path}")

        except ImportError:
            print("⚠️  matplotlib not available, skipping plots")
        except Exception as e:
            print(f"⚠️  Could not create plots: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate federated model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--experiment-dir", type=str, default=None,
                        help="Experiment directory containing checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--personalize-epochs", type=int, default=3,
                        help="Number of epochs to personalize head before evaluation")
    args = parser.parse_args()

    print("=" * 70)
    print("FEDERATED MODEL EVALUATION")
    print("=" * 70)

    evaluator = FederatedEvaluator(
        config_path=args.config,
        experiment_dir=args.experiment_dir,
    )

    model = evaluator.load_model(checkpoint_path=args.checkpoint)

    results = evaluator.evaluate_all_clients(
        model,
        personalize_epochs=args.personalize_epochs
    )

    if not results:
        print("\n❌ No results — check data availability")
        return

    report = evaluator.generate_report(results)

    evaluator.visualize_results(results)

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    overall = report.get("overall_metrics", {})
    print(f"Clients evaluated: {report['num_clients_evaluated']}")
    print(f"Mean Accuracy: {overall.get('mean_accuracy', 0):.4f} "
          f"(±{overall.get('std_accuracy', 0):.4f})")
    print(f"Mean Loss:     {overall.get('mean_loss', 0):.4f}")
    for k in ["mean_precision", "mean_recall", "mean_ndcg@10", "mean_mrr"]:
        if k in overall:
            print(f"{k.replace('mean_', 'Mean ').title()}: {overall[k]:.4f}")
    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()