"""
Convergence Analysis for Federated Learning
Tracks and visualizes training metrics over rounds
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json

class ConvergenceAnalyzer:
    """Analyze convergence of federated learning"""
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.metrics_history = []
    
    def load_training_history(self):
        """Load training metrics from all rounds"""
        # Load from experiment logs
        history_file = self.experiment_dir / "training_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                self.metrics_history = json.load(f)
        return self.metrics_history
    
    def plot_convergence(self, save_path: str = None):
        """
        Plot convergence curves
        - Training loss per round
        - Validation accuracy per round
        - Communication cost per round
        """
        if not self.metrics_history:
            print("No history loaded")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Federated Learning Convergence Analysis', fontsize=16)
        
        rounds = range(len(self.metrics_history))
        
        # 1. Training Loss
        train_loss = [m['train_loss'] for m in self.metrics_history]
        axes[0, 0].plot(rounds, train_loss, 'b-', linewidth=2, marker='o')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Training Loss')
        axes[0, 0].set_title('Training Loss vs Round')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Validation NDCG
        val_ndcg = [m['val_ndcg'] for m in self.metrics_history]
        axes[0, 1].plot(rounds, val_ndcg, 'g-', linewidth=2, marker='s')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('NDCG@10')
        axes[0, 1].set_title('Validation NDCG@10 vs Round')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0.85, color='r', linestyle='--', label='Target (0.85)')
        axes[0, 1].legend()
        
        # 3. Client Variance
        client_variance = [m.get('client_variance', 0) for m in self.metrics_history]
        axes[1, 0].plot(rounds, client_variance, 'r-', linewidth=2, marker='^')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Client Model Variance')
        axes[1, 0].set_title('Client Heterogeneity (lower = more converged)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Communication Cost (cumulative)
        comm_cost = [m.get('communication_mb', 0) for m in self.metrics_history]
        cumulative_cost = np.cumsum(comm_cost)
        axes[1, 1].plot(rounds, cumulative_cost, 'purple', linewidth=2, marker='d')
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('Cumulative Communication (MB)')
        axes[1, 1].set_title('Communication Overhead')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved convergence plot to {save_path}")
        
        plt.show()
    
    def calculate_convergence_metrics(self):
        """Calculate convergence statistics"""
        if not self.metrics_history:
            return None
        
        val_ndcg = [m['val_ndcg'] for m in self.metrics_history]
        
        # Find convergence round (when improvement < 1%)
        convergence_round = None
        for i in range(1, len(val_ndcg)):
            improvement = (val_ndcg[i] - val_ndcg[i-1]) / val_ndcg[i-1]
            if abs(improvement) < 0.01:  # < 1% improvement
                convergence_round = i
                break
        
        # Final performance
        final_ndcg = val_ndcg[-1]
        best_ndcg = max(val_ndcg)
        best_round = val_ndcg.index(best_ndcg)
        
        # Total communication cost
        total_comm_mb = sum([m.get('communication_mb', 0) for m in self.metrics_history])
        
        return {
            'convergence_round': convergence_round,
            'total_rounds': len(val_ndcg),
            'final_ndcg': final_ndcg,
            'best_ndcg': best_ndcg,
            'best_round': best_round,
            'total_communication_mb': total_comm_mb,
            'avg_communication_per_round_mb': total_comm_mb / len(val_ndcg)
        }
    
    def generate_convergence_report(self):
        """Generate text report"""
        metrics = self.calculate_convergence_metrics()
        
        if not metrics:
            return "No metrics available"
        
        report = f"""
# Convergence Analysis Report

## Key Findings

- **Convergence Round**: {metrics['convergence_round'] or 'Not converged'}
- **Total Rounds**: {metrics['total_rounds']}
- **Final NDCG@10**: {metrics['final_ndcg']:.4f}
- **Best NDCG@10**: {metrics['best_ndcg']:.4f} (at round {metrics['best_round']})

## Communication Efficiency

- **Total Communication**: {metrics['total_communication_mb']:.2f} MB
- **Avg per Round**: {metrics['avg_communication_per_round_mb']:.2f} MB

## Conclusion

Model {"converged" if metrics['convergence_round'] else "did not converge"} within {metrics['total_rounds']} rounds.
{"Early stopping could be applied at round " + str(metrics['convergence_round']) + "." if metrics['convergence_round'] else ""}
"""
        return report


def simulate_training_history(num_rounds: int = 20):
    """
    Simulate training history for demonstration
    In real scenario, this comes from actual training
    """
    history = []
    
    # Simulate realistic convergence
    base_loss = 2.5
    base_ndcg = 0.65
    
    for round_num in range(num_rounds):
        # Loss decreases with diminishing returns
        train_loss = base_loss * np.exp(-0.15 * round_num) + np.random.normal(0, 0.05)
        
        # NDCG increases with diminishing returns
        val_ndcg = base_ndcg + (0.85 - base_ndcg) * (1 - np.exp(-0.2 * round_num)) + np.random.normal(0, 0.01)
        
        # Client variance decreases (models become more similar)
        client_variance = 0.5 * np.exp(-0.1 * round_num) + np.random.normal(0, 0.02)
        
        # Communication cost (constant per round)
        communication_mb = 25 + np.random.normal(0, 2)
        
        history.append({
            'round': round_num,
            'train_loss': max(train_loss, 0.1),
            'val_ndcg': min(max(val_ndcg, 0), 1),
            'client_variance': max(client_variance, 0),
            'communication_mb': max(communication_mb, 0)
        })
    
    return history


if __name__ == "__main__":
    print("="*70)
    print("CONVERGENCE ANALYSIS")
    print("="*70)
    
    # Simulate training history (replace with real data)
    history = simulate_training_history(num_rounds=20)
    
    # Save simulated history
    output_dir = Path("experiments/federated_training")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Analyze
    analyzer = ConvergenceAnalyzer(str(output_dir))
    analyzer.metrics_history = history
    
    # Plot
    analyzer.plot_convergence(save_path=str(output_dir / "convergence_plot.png"))
    
    # Report
    report = analyzer.generate_convergence_report()
    print(report)
    
    # Save report
    with open(output_dir / "convergence_report.md", 'w') as f:
        f.write(report)
    
    print(f"\n✅ Results saved to {output_dir}")
