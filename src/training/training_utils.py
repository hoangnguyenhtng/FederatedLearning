"""
Training Utilities for Federated Multi-Modal Recommendation
Includes metrics, evaluation, and helper functions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time
from pathlib import Path


class MetricsCalculator:
    """Calculate recommendation metrics"""
    
    @staticmethod
    def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate accuracy"""
        pred_labels = predictions.argmax(dim=1)
        return accuracy_score(targets.cpu().numpy(), pred_labels.cpu().numpy())
    
    @staticmethod
    def precision_recall(predictions: torch.Tensor, targets: torch.Tensor, average='macro') -> Tuple[float, float]:
        """Calculate precision and recall"""
        pred_labels = predictions.argmax(dim=1)
        
        precision = precision_score(
            targets.cpu().numpy(),
            pred_labels.cpu().numpy(),
            average=average,
            zero_division=0
        )
        
        recall = recall_score(
            targets.cpu().numpy(),
            pred_labels.cpu().numpy(),
            average=average,
            zero_division=0
        )
        
        return precision, recall
    
    @staticmethod
    def ndcg_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
        """
        Calculate NDCG@K (Normalized Discounted Cumulative Gain)
        For rating prediction task
        """
        # Get predicted scores (use softmax probabilities weighted by rating values)
        probs = torch.softmax(predictions, dim=1)
        rating_values = torch.arange(1, 6, device=predictions.device).float()
        predicted_ratings = (probs * rating_values).sum(dim=1)
        
        # Convert targets (0-4) back to ratings (1-5)
        true_ratings = targets.float() + 1
        
        # Calculate DCG
        k = min(k, len(predicted_ratings))
        
        # Sort by predicted ratings
        sorted_indices = predicted_ratings.argsort(descending=True)[:k]
        sorted_true_ratings = true_ratings[sorted_indices]
        
        # DCG = sum(rel_i / log2(i+1))
        discounts = torch.log2(torch.arange(2, k + 2, device=predictions.device).float())
        dcg = (sorted_true_ratings / discounts).sum()
        
        # IDCG (ideal DCG)
        ideal_sorted_ratings = true_ratings.sort(descending=True)[0][:k]
        idcg = (ideal_sorted_ratings / discounts).sum()
        
        if idcg == 0:
            return 0.0
        
        return (dcg / idcg).item()
    
    @staticmethod
    def mrr(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calculate Mean Reciprocal Rank
        For each item, find rank of correct rating in predictions
        """
        # Get predicted rating (most probable)
        pred_labels = predictions.argmax(dim=1)
        
        # Calculate reciprocal rank
        reciprocal_ranks = []
        for pred, target in zip(pred_labels, targets):
            if pred == target:
                reciprocal_ranks.append(1.0)
            else:
                # Find rank of correct answer in sorted predictions
                sorted_preds = predictions[len(reciprocal_ranks)].argsort(descending=True)
                rank = (sorted_preds == target).nonzero(as_tuple=True)[0].item() + 1
                reciprocal_ranks.append(1.0 / rank)
        
        return np.mean(reciprocal_ranks)


def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    compute_all: bool = False
) -> Dict[str, float]:
    """
    Calculate evaluation metrics
    
    Args:
        predictions: Model predictions [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
        compute_all: Whether to compute all metrics (slower)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': MetricsCalculator.accuracy(predictions, targets)
    }
    
    if compute_all:
        precision, recall = MetricsCalculator.precision_recall(predictions, targets)
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['ndcg@10'] = MetricsCalculator.ndcg_at_k(predictions, targets, k=10)
        metrics['mrr'] = MetricsCalculator.mrr(predictions, targets)
    
    return metrics


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        """
        Check if should stop training
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


class TrainingLogger:
    """Log training metrics"""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'epoch_time': []
        }
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        epoch_time: float
    ):
        """Log metrics for one epoch"""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        self.history['epoch_time'].append(epoch_time)
        
        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.2f}s")
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get training history"""
        return self.history


def setup_logging(log_dir: str) -> None:
    """
    Setup logging directory
    
    Args:
        log_dir: Directory to save logs
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Logging directory: {log_path}")


def train_one_epoch(
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    text_encoder=None
) -> Tuple[float, float]:
    """
    Train model for one epoch
    
    Returns:
        average_loss, accuracy
    """
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    for batch in train_loader:
        # Move data to device
        user_ids = batch['user_id'].to(device)
        item_ids = batch['item_id'].to(device)
        image_features = batch['image_features'].to(device)
        behavior_features = batch['behavior_features'].to(device)
        # Use 'label' (rating-1, range 0-4) for rating prediction task
        # If 'label' not available, convert rating (1-5) to label (0-4)
        if 'label' in batch:
            targets = batch['label'].to(device)
        else:
            targets = (batch['rating'].to(device) - 1).clamp(0, 4)  # Convert 1-5 → 0-4
        
        # Encode text (if encoder provided)
        if text_encoder is not None:
            text_embeddings = text_encoder.encode(
                batch['text'],
                convert_to_tensor=True,
                device=device
            )
        else:
            # Use dummy embeddings
            text_embeddings = torch.randn(
                len(batch['text']), 384,
                device=device
            )
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(
            text_embeddings,
            image_features,
            behavior_features
        )
        
        # Calculate loss
        loss = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        all_predictions.append(predictions.detach())
        all_targets.append(targets.detach())
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(train_loader)
    
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    accuracy = MetricsCalculator.accuracy(all_predictions, all_targets)
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader,
    criterion: nn.Module,
    device: torch.device,
    text_encoder=None,
    compute_all_metrics: bool = False
) -> Dict[str, float]:
    """
    Evaluate model on test set
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        criterion: Loss function
        device: Device to use
        text_encoder: Text encoder (optional)
        compute_all_metrics: Whether to compute all metrics (slower)
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    for batch in test_loader:
        # Move data to device
        user_ids = batch['user_id'].to(device)
        item_ids = batch['item_id'].to(device)
        image_features = batch['image_features'].to(device)
        behavior_features = batch['behavior_features'].to(device)
        # Use 'label' (rating-1, range 0-4) for rating prediction task
        # If 'label' not available, convert rating (1-5) to label (0-4)
        if 'label' in batch:
            targets = batch['label'].to(device)
        else:
            targets = (batch['rating'].to(device) - 1).clamp(0, 4)  # Convert 1-5 → 0-4
        
        # Encode text
        if text_encoder is not None:
            text_embeddings = text_encoder.encode(
                batch['text'],
                convert_to_tensor=True,
                device=device
            )
        else:
            text_embeddings = torch.randn(
                len(batch['text']), 384,
                device=device
            )
        
        # Forward pass
        predictions = model(
            text_embeddings,
            image_features,
            behavior_features
        )
        
        # Calculate loss
        loss = criterion(predictions, targets)
        total_loss += loss.item()
        
        # Store predictions and targets
        all_predictions.append(predictions)
        all_targets.append(targets)
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    # Calculate metrics
    metrics = {
        'loss': total_loss / len(test_loader),
        'accuracy': MetricsCalculator.accuracy(all_predictions, all_targets)
    }
    
    if compute_all_metrics:
        precision, recall = MetricsCalculator.precision_recall(
            all_predictions, all_targets
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['ndcg@10'] = MetricsCalculator.ndcg_at_k(
            all_predictions, all_targets, k=10
        )
        metrics['mrr'] = MetricsCalculator.mrr(
            all_predictions, all_targets
        )
    
    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str
):
    """Save model checkpoint"""
    # Create directory if not exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, path)
    print(f"✅ Checkpoint saved to {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    device: torch.device
) -> int:
    """
    Load model checkpoint
    
    Args:
        model: Model to load into
        optimizer: Optimizer to load into (optional)
        path: Path to checkpoint
        device: Device to load to
    
    Returns:
        epoch number
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    print(f"✅ Checkpoint loaded from {path}")
    print(f"   Epoch: {epoch}, Metrics: {metrics}")
    
    return epoch


# Export all functions
__all__ = [
    'MetricsCalculator',
    'calculate_metrics',
    'EarlyStopping',
    'TrainingLogger',
    'setup_logging',
    'train_one_epoch',
    'evaluate',
    'save_checkpoint',
    'load_checkpoint'
]