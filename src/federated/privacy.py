"""
Differential Privacy Implementation for Federated Learning

Features:
1. DP-SGD optimizer (via Opacus)
2. Privacy budget tracking (epsilon, delta)
3. Gradient clipping
4. Noise injection
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import logging

# Try to import Opacus
try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    logging.warning("Opacus not available. Differential privacy will be disabled.")

logger = logging.getLogger(__name__)


def apply_differential_privacy(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    noise_multiplier: float = 1.1,
    max_grad_norm: float = 1.0,
    target_epsilon: float = 5.0,
    target_delta: Optional[float] = None,
    epochs: int = 3
) -> Tuple[nn.Module, torch.optim.Optimizer, Optional[PrivacyEngine]]:
    """
    Apply Differential Privacy to model training
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        data_loader: Training data loader
        noise_multiplier: Noise multiplier for DP-SGD (higher = more privacy, less accuracy)
        max_grad_norm: Maximum gradient norm for clipping
        target_epsilon: Target privacy budget (lower = more privacy)
        target_delta: Target delta (usually 1/n where n is dataset size)
        epochs: Number of training epochs
        
    Returns:
        model: Privacy-enabled model
        optimizer: DP optimizer
        privacy_engine: Privacy engine (None if Opacus not available)
    """
    if not OPACUS_AVAILABLE:
        logger.warning("Opacus not available. Returning original model and optimizer.")
        return model, optimizer, None
    
    try:
        # Make model compatible with Opacus
        model = ModuleValidator.fix(model)
        
        # Calculate delta if not provided
        if target_delta is None:
            # Delta = 1/n is a common choice
            dataset_size = len(data_loader.dataset)
            target_delta = 1.0 / dataset_size
        
        # Create Privacy Engine
        privacy_engine = PrivacyEngine()
        
        # Attach privacy engine to model
        model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            epochs=epochs,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_grad_norm=max_grad_norm
        )
        
        logger.info(f"Differential Privacy enabled:")
        logger.info(f"  - Target ε = {target_epsilon}")
        logger.info(f"  - Target δ = {target_delta:.2e}")
        logger.info(f"  - Noise multiplier = {noise_multiplier}")
        logger.info(f"  - Max grad norm = {max_grad_norm}")
        
        return model, optimizer, privacy_engine
        
    except Exception as e:
        logger.error(f"Failed to apply differential privacy: {e}")
        logger.warning("Falling back to non-private training")
        return model, optimizer, None


def get_dp_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    noise_multiplier: float = 1.1,
    max_grad_norm: float = 1.0
) -> Tuple[torch.optim.Optimizer, Optional[PrivacyEngine]]:
    """
    Get DP-enabled optimizer (simpler interface)
    
    Args:
        model: PyTorch model
        optimizer: Base optimizer
        data_loader: Data loader
        noise_multiplier: Noise multiplier
        max_grad_norm: Max gradient norm
        
    Returns:
        dp_optimizer: DP-enabled optimizer
        privacy_engine: Privacy engine
    """
    if not OPACUS_AVAILABLE:
        return optimizer, None
    
    try:
        model = ModuleValidator.fix(model)
        privacy_engine = PrivacyEngine()
        
        model, optimizer, data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm
        )
        
        return optimizer, privacy_engine
        
    except Exception as e:
        logger.error(f"Failed to create DP optimizer: {e}")
        return optimizer, None


def compute_privacy_budget(
    privacy_engine: Optional[PrivacyEngine],
    delta: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute current privacy budget (epsilon)
    
    Args:
        privacy_engine: Opacus privacy engine
        delta: Target delta
        
    Returns:
        Dict with epsilon and delta values
    """
    if privacy_engine is None or not OPACUS_AVAILABLE:
        return {
            'epsilon': float('inf'),
            'delta': 0.0,
            'available': False
        }
    
    try:
        if delta is None:
            delta = privacy_engine.target_delta
        
        epsilon = privacy_engine.get_epsilon(delta)
        
        return {
            'epsilon': epsilon,
            'delta': delta,
            'available': True
        }
        
    except Exception as e:
        logger.error(f"Failed to compute privacy budget: {e}")
        return {
            'epsilon': float('inf'),
            'delta': 0.0,
            'available': False
        }


def clip_gradients(model: nn.Module, max_grad_norm: float = 1.0):
    """
    Manually clip gradients (alternative to Opacus)
    
    Args:
        model: PyTorch model
        max_grad_norm: Maximum gradient norm
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)


def add_noise_to_gradients(model: nn.Module, noise_multiplier: float = 0.1):
    """
    Manually add Gaussian noise to gradients (alternative to Opacus)
    
    Args:
        model: PyTorch model
        noise_multiplier: Noise multiplier
    """
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * noise_multiplier
            param.grad += noise


class ManualDPOptimizer:
    """
    Manual DP optimizer (fallback when Opacus not available)
    """
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 noise_multiplier: float = 1.1,
                 max_grad_norm: float = 1.0):
        self.optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        
    def step(self, model: nn.Module):
        """Step with gradient clipping and noise"""
        clip_gradients(model, self.max_grad_norm)
        add_noise_to_gradients(model, self.noise_multiplier)
        self.optimizer.step()
        
    def zero_grad(self):
        """Zero gradients"""
        self.optimizer.zero_grad()


# Example usage
if __name__ == "__main__":
    # Test privacy module
    from torch.utils.data import DataLoader, TensorDataset
    
    print("Testing Privacy Module")
    print(f"Opacus available: {OPACUS_AVAILABLE}")
    
    # Create dummy model and data
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=32)
    
    # Test DP application
    model, optimizer, privacy_engine = apply_differential_privacy(
        model=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=1.1,
        max_grad_norm=1.0,
        target_epsilon=5.0,
        epochs=3
    )
    
    # Test privacy budget computation
    budget = compute_privacy_budget(privacy_engine)
    print(f"\nPrivacy Budget: {budget}")
    
    print("\n✓ Privacy module test completed")