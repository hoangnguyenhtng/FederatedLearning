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
from typing import Dict, List
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from models.recommendation_model import FedPerRecommender
from data_generation.federated_dataloader import FederatedDataLoader
from training_utils import evaluate, MetricsCalculator


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
    
    def load_model(self, checkpoint_path: str) -> nn.Module:
        """Load trained model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model
        model_config = self.config['model']
        model = FedPerRecommender(
            text_dim=model_config['text']['embedding_dim'],
            image_dim=model_config['image']['output_dim'],
            behavior_dim=model_config['behavior']['output_dim'],
            hidden_dims=model_config['recommendation']['hidden_dims'],
            num_classes=5,
            num_users=1000,
            num_items=10000,
            shared_layers=self.config['federated']['shared_layers'],
            personal_layers=self.config['federated']['personal_layers']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
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
        
        for client_id in client_ids:
            # Load client data
            data_loader = FederatedDataLoader(
                client_id=client_id,
                data_dir='./data/simulated_clients',
                batch_size=self.config['data']['batch_size'],
                test_split=self.config['data']['test_split']
            )
            
            _, test_loader = data_loader.create_dataloaders()
            
            # Evaluate
            metrics = evaluate(
                model=model,
                test_loader=test_loader,
                criterion=nn.CrossEntropyLoss(),
                device=self.device,
                text_encoder=None,
                compute_all_metrics=True
            )
            
            # Get client metadata
            metadata = data_loader.metadata
            
            result = {
                'client_id': client_id,
                'num_users': metadata['num_users'],
                'num_interactions': metadata['num_interactions'],
                'preference_distribution': metadata['preference_distribution'],
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
    print("="*70)
    print("FEDERATED MODEL EVALUATION")
    print("="*70)
    
    # Configuration
    experiment_dir = './experiments/fedper_multimodal_v1'
    checkpoint_path = f'{experiment_dir}/models/global_model_round_50.pt'
    
    # Create evaluator
    evaluator = FederatedEvaluator(
        experiment_dir=experiment_dir,
        config_path='./configs/config.yaml'
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