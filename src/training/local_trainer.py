"""
Local Trainer for Federated Clients
Each client trains their model locally on their data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
from typing import Dict, Optional
import time
import os

# Fix: Add project root to path properly
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Now import with proper paths
from src.data_generation.federated_dataloader import FederatedDataLoader
from src.models.recommendation_model import FedPerRecommender
from src.training.training_utils import (
    train_one_epoch,
    evaluate,
    save_checkpoint,
    load_checkpoint,
    EarlyStopping,
    TrainingLogger
)


class LocalTrainer:
    """Local trainer for a federated client"""
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        config: Dict,
        device: torch.device = None,
        text_encoder = None
    ):
        """
        Initialize local trainer
        
        Args:
            client_id: Client identifier
            model: FedPerRecommender model
            config: Training configuration
            device: Device to train on
            text_encoder: Pre-trained text encoder
        """
        self.client_id = client_id
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_encoder = text_encoder
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Loss function
        class_counts = torch.tensor([0.04, 1.5, 15.9, 50.6, 31.7])  # From your data
        class_weights = 100.0 / class_counts  # Inverse weights
        class_weights = class_weights / class_weights.sum() * 5  # Normalize

        print(f"[Client {self.client_id}] Using class weights: {class_weights}")

        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 5),
            min_delta=config.get('min_delta', 0.001)
        )
        
        # Logger
        self.logger = TrainingLogger()
        
        # Load data
        data_dir = config.get('data_dir', str(PROJECT_ROOT / 'data' / 'simulated_clients'))
        self.data_loader = FederatedDataLoader(
            client_id=client_id,
            data_dir=data_dir,
            batch_size=config.get('batch_size', 32),
            test_split=config.get('test_split', 0.2),
            seed=config.get('seed', 42)
        )
        
        self.train_loader, self.test_loader = self.data_loader.create_dataloaders(
            text_encoder=self.text_encoder
        )
        
        print(f"\n✅ LocalTrainer initialized for Client {client_id}")
        print(f"   Device: {self.device}")
        print(f"   Train batches: {len(self.train_loader)}")
        print(f"   Test batches: {len(self.test_loader)}")
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer"""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 0.001)
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=self.config.get('momentum', 0.9)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def train_local_epochs(
        self,
        num_epochs: int,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train model for specified number of local epochs
        
        Args:
            num_epochs: Number of epochs to train
            verbose: Whether to print progress
        
        Returns:
            Dictionary of final metrics
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training Client {self.client_id} for {num_epochs} epochs")
            print(f"{'='*60}")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            
            # Train one epoch
            train_loss, train_acc = train_one_epoch(
                model=self.model,
                train_loader=self.train_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                device=self.device,
                text_encoder=self.text_encoder
            )
            
            # Evaluate
            val_metrics = evaluate(
                model=self.model,
                test_loader=self.test_loader,
                criterion=self.criterion,
                device=self.device,
                text_encoder=self.text_encoder,
                compute_all_metrics=False
            )
            
            epoch_time = time.time() - epoch_start_time
            
            # Log metrics
            if verbose:
                self.logger.log_epoch(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_metrics['loss'],
                    train_acc=train_acc,
                    val_acc=val_metrics['accuracy'],
                    epoch_time=epoch_time
                )
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_model(f'best_client_{self.client_id}.pt')
            
            # Early stopping check
            if self.early_stopping(val_metrics['loss']):
                if verbose:
                    print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
                break
        
        # Final evaluation with all metrics
        final_metrics = evaluate(
            model=self.model,
            test_loader=self.test_loader,
            criterion=self.criterion,
            device=self.device,
            text_encoder=self.text_encoder,
            compute_all_metrics=True
        )
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Client {self.client_id} Training Complete")
            print(f"{'='*60}")
            print(f"Final Metrics:")
            for metric, value in final_metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        return final_metrics
    
    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Get model parameters (for federated aggregation)"""
        return {
            name: param.detach().cpu().clone()
            for name, param in self.model.named_parameters()
        }
    
    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set model parameters (from federated aggregation)"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in parameters:
                    param.copy_(parameters[name].to(self.device))
    
    def get_shared_parameters(self) -> Dict[str, torch.Tensor]:
        """Get only shared parameters (for FedPer)"""
        shared_params = {}
        for name, param in self.model.named_parameters():
            # Only include shared layers (base model)
            if 'personal_head' not in name:
                shared_params[name] = param.detach().cpu().clone()
        return shared_params
    
    def set_shared_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set only shared parameters (for FedPer)"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'personal_head' not in name and name in parameters:
                    param.copy_(parameters[name].to(self.device))
    
    def save_model(self, filename: str):
        """Save model checkpoint"""
        save_dir = Path(self.config.get('save_dir', str(PROJECT_ROOT / 'models' / 'checkpoints')))
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / filename
        
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=0,  # Will be updated in federated training
            metrics={},
            path=str(save_path)
        )
    
    def load_model(self, filename: str):
        """Load model checkpoint"""
        load_path = Path(self.config.get('save_dir', str(PROJECT_ROOT / 'models' / 'checkpoints'))) / filename
        
        if load_path.exists():
            load_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                path=str(load_path),
                device=self.device
            )
        else:
            print(f"⚠️  Checkpoint not found: {load_path}")
    
    def get_training_history(self) -> Dict:
        """Get training history"""
        return self.logger.get_history()


def test_local_trainer():
    """Test local trainer"""
    print("=" * 70)
    print("Testing Local Trainer")
    print("=" * 70)
    
    # Load config
    import yaml
    config_path = PROJECT_ROOT / 'configs' / 'config.yaml'
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        print(f"   Please create config.yaml first")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model_config = config['model']
    
    model = FedPerRecommender(
        text_dim=model_config['text']['embedding_dim'],
        image_dim=model_config['image']['output_dim'],
        behavior_dim=model_config['behavior']['output_dim'],
        hidden_dims=model_config['recommendation']['hidden_dims'],
        num_classes=5,  # 5 rating classes
        num_users=100,  # Dummy
        num_items=1000,  # Dummy
        shared_layers=config['federated']['shared_layers'],
        personal_layers=config['federated']['personal_layers']
    )
    
    # Training config
    train_config = {
        'batch_size': 16,
        'test_split': 0.2,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'patience': 3,
        'data_dir': str(PROJECT_ROOT / 'data' / 'simulated_clients'),
        'save_dir': str(PROJECT_ROOT / 'models' / 'checkpoints')
    }
    
    # Check if data exists
    data_dir = Path(train_config['data_dir'])
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print(f"   Please run data generation first:")
        print(f"   cd src/data_generation && python main_data_generation.py")
        return
    
    # Create trainer
    try:
        trainer = LocalTrainer(
            client_id=0,
            model=model,
            config=train_config,
            device=torch.device('cpu')
        )
        
        # Train for 2 epochs
        metrics = trainer.train_local_epochs(num_epochs=2, verbose=True)
        
        print("\n" + "=" * 70)
        print("✅ Local trainer test complete!")
        print("=" * 70)
        print(f"Final metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_local_trainer()