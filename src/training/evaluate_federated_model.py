"""
Evaluation Script for Federated Multi-Modal Recommendation
Compare personalized models vs baseline
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import yaml
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.models.recommendation_model import FedPerRecommender
from src.training.training_utils import evaluate, MetricsCalculator, resolve_amazon_federated_data_dir


class FederatedEvaluator:
    """Evaluate federated model performance"""
    
    def __init__(
        self,
        experiment_dir: str,
        config_path: str = './configs/config.yaml'
    ):
        """
        Initialize evaluator
        
        Args:
            experiment_dir: Directory containing experiment results
            config_path: Path to configuration file
        """
        self.experiment_dir = Path(experiment_dir)
        self.config_path = config_path
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"📊 Evaluator initialized")
        print(f"   Experiment: {experiment_dir}")
        print(f"   Device: {self.device}")
    
    def _load_federated_dataloaders(self) -> Dict[int, tuple]:
        """
        Giống logic trong federated_training_pipeline._load_data:
        Amazon nếu có, không thì synthetic — để evaluation khớp dữ liệu đã train.
        """
        from src.data_generation.federated_dataloader import get_federated_dataloaders
        
        num_clients = self.config['federated']['num_clients']
        batch_size = self.config['training']['batch_size']
        test_split = self.config['training'].get('test_split', 0.2)
        paths_cfg = self.config.get('paths') or {}
        amazon_dir = resolve_amazon_federated_data_dir(self.config, cwd=project_root)
        synthetic_dir = (project_root / paths_cfg.get('data_dir', 'data') / 'simulated_clients').resolve()

        if amazon_dir is not None:
            print(f"📂 Evaluation data: processed Amazon pickles → {amazon_dir}")
            from src.data_generation.amazon_dataloader import get_amazon_dataloaders
            return get_amazon_dataloaders(
                num_clients=num_clients,
                data_dir=str(amazon_dir),
                batch_size=batch_size,
                test_split=test_split,
            )
        if synthetic_dir.exists():
            print("📂 Evaluation data: simulated_clients")
            loaders_list = get_federated_dataloaders(
                num_clients=num_clients,
                data_dir=str(synthetic_dir),
                batch_size=batch_size,
                test_split=test_split,
            )
            out = {}
            for client_id, loaders in enumerate(loaders_list):
                if loaders and len(loaders) == 2:
                    tr, te = loaders
                    if tr is not None and te is not None:
                        out[client_id] = (tr, te)
            return out
        raise FileNotFoundError(
            f"No data for evaluation. Expected Amazon at {amazon_dir} or synthetic at {synthetic_dir}"
        )
    
    def load_model(self, checkpoint_path: str) -> nn.Module:
        """Load trained model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model using current config structure
        model_config = self.config['model']
        from src.models.multimodal_encoder import MultiModalEncoder
        from src.models.recommendation_model import FedPerRecommender
        
        # Create multimodal encoder (tên tham số khớp multimodal_encoder.py)
        multimodal_encoder = MultiModalEncoder(
            text_dim=model_config.get('text_embedding_dim', 384),
            image_dim=model_config.get('image_embedding_dim', 2048),
            behavior_dim=model_config.get('behavior_embedding_dim', 32),
            hidden_dim=model_config.get('hidden_dim', 256),
            output_dim=384,
        )
        
        # Create FedPerRecommender
        model = FedPerRecommender(
            multimodal_encoder=multimodal_encoder,
            shared_hidden_dims=model_config['shared_hidden_dims'],
            personal_hidden_dims=model_config['personal_hidden_dims'],
            num_classes=model_config['num_classes'],
            dropout=model_config.get('dropout', 0.2)
        )
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def evaluate_all_clients(
        self,
        model: nn.Module,
        client_ids: List[int] = None
    ) -> pd.DataFrame:
        """
        Evaluate model on all clients
        
        Args:
            model: Global model
            client_ids: List of client IDs (default: all)
        
        Returns:
            DataFrame with per-client metrics
        """
        if client_ids is None:
            client_ids = list(range(self.config['federated']['num_clients']))
        
        print(f"\n📊 Evaluating {len(client_ids)} clients...")
        
        results = []
        
        dataloaders = self._load_federated_dataloaders()
        
        for client_id in client_ids:
            if client_id not in dataloaders:
                print(f"⚠️  Client {client_id} not found, skipping...")
                continue
            
            train_loader, test_loader = dataloaders[client_id]
            
            # Evaluate
            metrics = evaluate(
                model=model,
                test_loader=test_loader,
                criterion=nn.CrossEntropyLoss(),
                device=self.device,
                text_encoder=None,
                compute_all_metrics=True
            )
            
            result = {
                'client_id': client_id,
                **metrics
            }
            
            results.append(result)
            
            print(f"  Client {client_id}: Acc={metrics['accuracy']:.4f}, "
                  f"NDCG@10={metrics['ndcg@10']:.4f}")
        
        return pd.DataFrame(results)
    
    def evaluate_by_preference_type(
        self,
        results_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate metrics by user preference type
        
        Args:
            results_df: Results from evaluate_all_clients
        
        Returns:
            Aggregated metrics by preference type
        """
        print("\n📊 Aggregating by preference type...")
        
        if 'preference_distribution' not in results_df.columns:
            print("⚠️  Không có cột preference_distribution (chưa join metadata client). Bỏ qua nhóm preference.")
            return pd.DataFrame()
        
        # Extract dominant preference for each client
        preference_data = []
        
        for _, row in results_df.iterrows():
            pref_dist = row['preference_distribution']
            
            # Find dominant preference type
            if isinstance(pref_dist, str):
                pref_dist = eval(pref_dist)
            
            dominant_pref = max(pref_dist.items(), key=lambda x: x[1])[0]
            dominant_count = pref_dist[dominant_pref]
            total_users = sum(pref_dist.values())
            dominant_ratio = dominant_count / total_users if total_users > 0 else 0
            
            # Only consider if dominant preference > 50%
            if dominant_ratio > 0.5:
                preference_data.append({
                    'preference_type': dominant_pref,
                    'accuracy': row['accuracy'],
                    'precision': row['precision'],
                    'recall': row['recall'],
                    'ndcg@10': row['ndcg@10'],
                    'mrr': row['mrr'],
                    'num_users': row['num_users']
                })
        
        pref_df = pd.DataFrame(preference_data)
        
        # Group by preference type
        aggregated = pref_df.groupby('preference_type').agg({
            'accuracy': 'mean',
            'precision': 'mean',
            'recall': 'mean',
            'ndcg@10': 'mean',
            'mrr': 'mean',
            'num_users': 'sum'
        }).reset_index()
        
        print(aggregated)
        
        return aggregated
    
    def visualize_results(
        self,
        results_df: pd.DataFrame,
        preference_df: pd.DataFrame,
        save_dir: str = None
    ):
        """Create visualization of results"""
        if save_dir is None:
            save_dir = self.experiment_dir / 'evaluation'
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Per-client accuracy
        axes[0, 0].bar(results_df['client_id'], results_df['accuracy'])
        axes[0, 0].set_xlabel('Client ID')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Per-Client Accuracy')
        axes[0, 0].axhline(
            y=results_df['accuracy'].mean(),
            color='r',
            linestyle='--',
            label=f'Mean: {results_df["accuracy"].mean():.3f}'
        )
        axes[0, 0].legend()
        
        # 2. Metrics by preference type
        if len(preference_df) > 0:
            x = np.arange(len(preference_df))
            width = 0.2
            
            axes[0, 1].bar(x - width, preference_df['accuracy'], width, label='Accuracy')
            axes[0, 1].bar(x, preference_df['precision'], width, label='Precision')
            axes[0, 1].bar(x + width, preference_df['recall'], width, label='Recall')
            
            axes[0, 1].set_xlabel('Preference Type')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].set_title('Metrics by Preference Type')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(preference_df['preference_type'], rotation=45)
            axes[0, 1].legend()
        else:
            axes[0, 1].text(0.5, 0.5, 'No preference metadata\n(join users.csv in future)', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Metrics by Preference Type')
        
        # 3. NDCG@10 comparison
        axes[1, 0].bar(results_df['client_id'], results_df['ndcg@10'])
        axes[1, 0].set_xlabel('Client ID')
        axes[1, 0].set_ylabel('NDCG@10')
        axes[1, 0].set_title('NDCG@10 per Client')
        axes[1, 0].axhline(
            y=results_df['ndcg@10'].mean(),
            color='r',
            linestyle='--',
            label=f'Mean: {results_df["ndcg@10"].mean():.3f}'
        )
        axes[1, 0].legend()
        
        # 4. Distribution of metrics
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'ndcg@10']
        data_to_plot = [results_df[m].values for m in metrics_to_plot]
        
        axes[1, 1].boxplot(data_to_plot, labels=metrics_to_plot)
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Distribution of Metrics Across Clients')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        viz_path = save_path / 'evaluation_results.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Visualization saved to {viz_path}")
        plt.close()
    
    def generate_report(
        self,
        results_df: pd.DataFrame,
        preference_df: pd.DataFrame,
        save_dir: str = None
    ):
        """Generate evaluation report"""
        if save_dir is None:
            save_dir = self.experiment_dir / 'evaluation'
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        report = {
            'overall_metrics': {
                'mean_accuracy': float(results_df['accuracy'].mean()),
                'std_accuracy': float(results_df['accuracy'].std()),
                'mean_precision': float(results_df['precision'].mean()),
                'mean_recall': float(results_df['recall'].mean()),
                'mean_ndcg@10': float(results_df['ndcg@10'].mean()),
                'mean_mrr': float(results_df['mrr'].mean())
            },
            'per_preference_type': preference_df.to_dict('records') if len(preference_df) > 0 else [],
            'per_client': results_df.to_dict('records')
        }
        
        # Save report
        report_path = save_path / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save CSV
        results_df.to_csv(save_path / 'client_results.csv', index=False)
        if len(preference_df) > 0:
            preference_df.to_csv(save_path / 'preference_results.csv', index=False)
        
        print(f"\n📝 Report saved to {save_path}")
        print(f"   - evaluation_report.json")
        print(f"   - client_results.csv")
        if len(preference_df) > 0:
            print(f"   - preference_results.csv")
        
        return report


def main():
    """Main evaluation script"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate federated checkpoint")
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="./experiments/fedper_multimodal_v1",
        help="Thư mục experiment (models/global_model_final.pt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/config.yaml",
        help="YAML trùng với train (num_clients, data paths, …)",
    )
    args = parser.parse_args()

    print("="*70)
    print("FEDERATED MODEL EVALUATION")
    print("="*70)
    
    # Configuration
    experiment_dir = args.experiment_dir
    checkpoint_path = f'{experiment_dir}/models/global_model_final.pt'
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("   Looking for alternative checkpoints...")
        models_dir = Path(experiment_dir) / "models"
        if models_dir.exists():
            checkpoints = list(models_dir.glob("*.pt"))
            if checkpoints:
                checkpoint_path = str(checkpoints[0])
                print(f"   Using: {checkpoint_path}")
            else:
                print("❌ No checkpoints found!")
                return
        else:
            print("❌ Models directory not found!")
            return
    
    # Create evaluator
    evaluator = FederatedEvaluator(
        experiment_dir=experiment_dir,
        config_path=args.config,
    )
    
    # Load model
    print(f"\n📦 Loading model from {checkpoint_path}...")
    model = evaluator.load_model(checkpoint_path)
    
    # Evaluate all clients
    results_df = evaluator.evaluate_all_clients(model)
    
    # Aggregate by preference type
    preference_df = evaluator.evaluate_by_preference_type(results_df)
    
    # Visualize results
    evaluator.visualize_results(results_df, preference_df)
    
    # Generate report
    report = evaluator.generate_report(results_df, preference_df)
    
    print("\n"+"="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Mean Accuracy: {report['overall_metrics']['mean_accuracy']:.4f} "
          f"(±{report['overall_metrics']['std_accuracy']:.4f})")
    print(f"Mean NDCG@10: {report['overall_metrics']['mean_ndcg@10']:.4f}")
    print(f"Mean MRR: {report['overall_metrics']['mean_mrr']:.4f}")
    
    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()