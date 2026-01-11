"""
Training Package for Federated Multi-Modal Recommendation

Components:
- training_utils: Metrics, evaluation, checkpointing
- local_trainer: Local training for each client
- federated_training_pipeline: Complete federated training orchestration
- evaluate_federated_model: Model evaluation and visualization
"""

from .training_utils import (
    MetricsCalculator,
    calculate_metrics,
    EarlyStopping,
    TrainingLogger,
    setup_logging,
    train_one_epoch,
    evaluate,
    save_checkpoint,
    load_checkpoint
)

# Conditional imports to avoid circular dependencies
try:
    from .local_trainer import LocalTrainer
except ImportError:
    LocalTrainer = None

try:
    from .federated_training_pipeline import FederatedTrainingPipeline
except ImportError:
    FederatedTrainingPipeline = None

__all__ = [
    # Training utilities
    'MetricsCalculator',
    'calculate_metrics',
    'EarlyStopping',
    'TrainingLogger',
    'setup_logging',
    'train_one_epoch',
    'evaluate',
    'save_checkpoint',
    'load_checkpoint',
    
    # Trainer
    'LocalTrainer',
    
    # Pipeline
    'FederatedTrainingPipeline'
]