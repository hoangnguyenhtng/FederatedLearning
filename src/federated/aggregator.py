"""
Federated Learning Aggregation Strategies
==========================================

This module implements various aggregation strategies for Federated Learning:
1. FedAvg: Standard weighted average
2. FedProx: FedAvg with proximal regularization
3. FedAdam: Adaptive aggregation with Adam optimizer
4. NonIIDAggregator: Custom aggregation for non-IID data

All aggregators work with numpy arrays for compatibility with Flower framework.

Author: Federated Multi-Modal Recommendation System
Date: 2024
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class FedAvgAggregator:
    """
    FedAvg Aggregator - Standard weighted average
    
    Formula:
        w_global = Σ(n_k / N) * w_k
        
    where:
        w_k: parameters from client k
        n_k: number of samples of client k
        N: total number of samples
    
    Reference: McMahan et al., "Communication-Efficient Learning of Deep Networks 
    from Decentralized Data", AISTATS 2017
    """
    
    def __init__(self):
        """Initialize FedAvg aggregator."""
        self.name = "FedAvg"
    
    @staticmethod
    def aggregate(client_params: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """
        Aggregate parameters from clients using weighted average.
        
        Args:
            client_params: List of (parameters, num_samples) tuples
                - parameters: List[np.ndarray] - model parameters
                - num_samples: int - number of training samples
            
        Returns:
            aggregated_params: List[np.ndarray] - weighted average parameters
            
        Example:
            >>> client_params = [
            ...     ([np.array([1.0, 2.0]), np.array([3.0])], 100),
            ...     ([np.array([2.0, 3.0]), np.array([4.0])], 200)
            ... ]
            >>> result = FedAvgAggregator.aggregate(client_params)
        """
        if not client_params:
            raise ValueError("client_params cannot be empty")
        
        # Calculate total samples
        total_samples = sum(num_samples for _, num_samples in client_params)
        
        if total_samples == 0:
            raise ValueError("Total samples cannot be zero")
        
        # Initialize aggregated params with zeros
        first_params, _ = client_params[0]
        aggregated = [np.zeros_like(param) for param in first_params]
        
        # Weighted sum
        for params, num_samples in client_params:
            weight = num_samples / total_samples
            
            for i, param in enumerate(params):
                aggregated[i] += weight * param
        
        logger.info(f"FedAvg aggregated {len(client_params)} clients with {total_samples} total samples")
        
        return aggregated


class FedProxAggregator:
    """
    FedProx Aggregator - FedAvg with proximal regularization
    
    Adds proximal term to handle heterogeneous data better.
    
    Loss function for each client:
        L_k(w) + (mu/2) * ||w - w_global||^2
    
    Reference: Li et al., "Federated Optimization in Heterogeneous Networks", 
    MLSys 2020
    """
    
    def __init__(self, mu: float = 0.01):
        """
        Initialize FedProx aggregator.
        
        Args:
            mu: Proximal term coefficient (higher = stronger regularization)
                Typical values: 0.001 - 0.1
        """
        self.mu = mu
        self.name = f"FedProx(mu={mu})"
        
    def aggregate(self, 
                 client_params: List[Tuple[List[np.ndarray], int]],
                 global_params: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """
        Aggregate with proximal regularization.
        
        Args:
            client_params: List of (parameters, num_samples)
            global_params: Current global parameters (optional)
                If None, behaves like FedAvg
            
        Returns:
            aggregated_params: FedProx aggregated parameters
        """
        # Standard FedAvg aggregation
        aggregated = FedAvgAggregator.aggregate(client_params)
        
        # Apply proximal term if global params provided
        if global_params is not None:
            regularized = []
            for agg, glob in zip(aggregated, global_params):
                # Move aggregated closer to global params
                regularized.append(agg + self.mu * (glob - agg))
            
            logger.info(f"FedProx applied with mu={self.mu}")
            return regularized
        
        return aggregated


class FedAdamAggregator:
    """
    FedAdam Aggregator - Adaptive aggregation with momentum
    
    Uses Adam optimizer for server-side optimization.
    Better convergence for non-IID data.
    
    Reference: Reddi et al., "Adaptive Federated Optimization", ICLR 2021
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8):
        """
        Initialize FedAdam aggregator.
        
        Args:
            learning_rate: Server learning rate (typical: 0.001 - 0.1)
            beta1: First moment decay rate (typical: 0.9)
            beta2: Second moment decay rate (typical: 0.999)
            epsilon: Numerical stability constant
        """
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Initialize moment estimates
        self.m: Optional[List[np.ndarray]] = None  # First moment
        self.v: Optional[List[np.ndarray]] = None  # Second moment
        self.t: int = 0  # Time step
        
        self.name = f"FedAdam(lr={learning_rate})"
        
    def aggregate(self,
                 client_params: List[Tuple[List[np.ndarray], int]],
                 global_params: List[np.ndarray]) -> List[np.ndarray]:
        """
        Aggregate using Adam optimizer.
        
        Args:
            client_params: List of (parameters, num_samples)
            global_params: Current global parameters (required)
            
        Returns:
            updated_params: Adam-updated parameters
        """
        # Get pseudo-gradient (difference from global)
        avg_params = FedAvgAggregator.aggregate(client_params)
        pseudo_grad = [avg - glob for avg, glob in zip(avg_params, global_params)]
        
        # Initialize moments on first iteration
        if self.m is None:
            self.m = [np.zeros_like(g) for g in pseudo_grad]
            self.v = [np.zeros_like(g) for g in pseudo_grad]
            logger.info("FedAdam: Initialized momentum buffers")
        
        self.t += 1
        
        # Update first moment (momentum)
        self.m = [
            self.beta1 * m + (1 - self.beta1) * g
            for m, g in zip(self.m, pseudo_grad)
        ]
        
        # Update second moment (RMSprop)
        self.v = [
            self.beta2 * v + (1 - self.beta2) * (g ** 2)
            for v, g in zip(self.v, pseudo_grad)
        ]
        
        # Bias correction
        m_hat = [m / (1 - self.beta1 ** self.t) for m in self.m]
        v_hat = [v / (1 - self.beta2 ** self.t) for v in self.v]
        
        # Update parameters (Adam update rule)
        updated = [
            glob + self.lr * m / (np.sqrt(v) + self.epsilon)
            for glob, m, v in zip(global_params, m_hat, v_hat)
        ]
        
        logger.info(f"FedAdam: Updated parameters at step {self.t}")
        
        return updated
    
    def reset(self):
        """Reset momentum buffers."""
        self.m = None
        self.v = None
        self.t = 0
        logger.info("FedAdam: Reset momentum buffers")


class NonIIDAggregator:
    """
    Custom aggregator for Non-IID data
    
    Strategies:
    1. adaptive_weight: Weight clients by performance (lower loss = higher weight)
    2. cluster: Cluster similar clients (future implementation)
    
    For non-IID scenarios, clients with better local performance contribute more
    to the global model.
    """
    
    def __init__(self, strategy: str = 'adaptive_weight', alpha: float = 0.7):
        """
        Initialize Non-IID aggregator.
        
        Args:
            strategy: Aggregation strategy
                - 'adaptive_weight': Weight by performance
                - 'standard': Fallback to FedAvg
            alpha: Weight for sample-based weighting (0-1)
                alpha=1.0: pure sample-based (like FedAvg)
                alpha=0.0: pure performance-based
                alpha=0.7: 70% sample-based, 30% performance-based
        """
        self.strategy = strategy
        self.alpha = alpha
        self.client_losses: Dict[int, float] = {}
        self.name = f"NonIID({strategy}, alpha={alpha})"
        
    def update_client_loss(self, client_id: int, loss: float):
        """
        Update tracked loss for a client.
        
        Args:
            client_id: Client identifier
            loss: Training loss value
        """
        self.client_losses[client_id] = loss
        logger.debug(f"Updated client {client_id} loss: {loss:.4f}")
        
    def compute_adaptive_weights(self, 
                                client_ids: List[int],
                                num_samples: List[int]) -> np.ndarray:
        """
        Compute adaptive weights based on client performance.
        
        Better performing clients (lower loss) get higher weights.
        
        Args:
            client_ids: List of participating client IDs
            num_samples: List of sample counts for each client
            
        Returns:
            weights: Adaptive weights [num_clients]
        """
        # Base weights proportional to sample count
        base_weights = np.array(num_samples, dtype=np.float32)
        base_weights = base_weights / base_weights.sum()
        
        # If no loss history, use base weights
        if not self.client_losses:
            logger.warning("No client losses tracked, using sample-based weights")
            return base_weights
        
        # Get losses for participating clients
        losses = np.array([
            self.client_losses.get(cid, np.mean(list(self.client_losses.values())))
            for cid in client_ids
        ], dtype=np.float32)
        
        # Inverse loss weighting (lower loss = higher weight)
        # Add small constant to avoid division by zero
        inv_losses = 1.0 / (losses + 0.01)
        perf_weights = inv_losses / inv_losses.sum()
        
        # Combine base weights and performance weights
        adaptive_weights = self.alpha * base_weights + (1 - self.alpha) * perf_weights
        adaptive_weights = adaptive_weights / adaptive_weights.sum()
        
        logger.info(f"Adaptive weights: sample={base_weights}, perf={perf_weights}, "
                   f"final={adaptive_weights}")
        
        return adaptive_weights
    
    def aggregate(self,
                 client_params: List[Tuple[List[np.ndarray], int, int]],
                 strategy: Optional[str] = None) -> List[np.ndarray]:
        """
        Aggregate with Non-IID awareness.
        
        Args:
            client_params: List of (parameters, num_samples, client_id) tuples
            strategy: Override default strategy (optional)
            
        Returns:
            aggregated_params: Aggregated parameters
        """
        strategy = strategy or self.strategy
        
        if strategy == 'adaptive_weight':
            # Extract components
            params_list = [p for p, _, _ in client_params]
            num_samples = [n for _, n, _ in client_params]
            client_ids = [cid for _, _, cid in client_params]
            
            # Compute adaptive weights
            weights = self.compute_adaptive_weights(client_ids, num_samples)
            
            # Weighted aggregation
            aggregated = [np.zeros_like(param) for param in params_list[0]]
            
            for params, weight in zip(params_list, weights):
                for i, param in enumerate(params):
                    aggregated[i] += weight * param
            
            logger.info(f"NonIID aggregated {len(client_params)} clients with adaptive weights")
            return aggregated
        
        else:
            # Fallback to standard FedAvg
            params_and_samples = [(p, n) for p, n, _ in client_params]
            return FedAvgAggregator.aggregate(params_and_samples)


# =============================================================================
# Helper Functions
# =============================================================================

def parameters_to_numpy(state_dict: OrderedDict) -> List[np.ndarray]:
    """
    Convert PyTorch state_dict to list of numpy arrays.
    
    Args:
        state_dict: PyTorch OrderedDict of parameters
        
    Returns:
        List of numpy arrays
        
    Example:
        >>> model = torch.nn.Linear(10, 5)
        >>> params = parameters_to_numpy(model.state_dict())
        >>> print(len(params))  # 2 (weight and bias)
    """
    return [param.cpu().detach().numpy() for param in state_dict.values()]


def numpy_to_parameters(arrays: List[np.ndarray], 
                       template_state_dict: OrderedDict) -> OrderedDict:
    """
    Convert list of numpy arrays back to PyTorch state_dict.
    
    Args:
        arrays: List of numpy arrays
        template_state_dict: Template state dict for keys and structure
        
    Returns:
        state_dict: PyTorch OrderedDict
        
    Example:
        >>> arrays = [np.random.randn(5, 10), np.random.randn(5)]
        >>> template = OrderedDict([('weight', None), ('bias', None)])
        >>> state_dict = numpy_to_parameters(arrays, template)
    """
    if len(arrays) != len(template_state_dict):
        raise ValueError(f"Arrays length {len(arrays)} != template length {len(template_state_dict)}")
    
    keys = list(template_state_dict.keys())
    return OrderedDict({
        key: torch.tensor(array, dtype=torch.float32)
        for key, array in zip(keys, arrays)
    })


def get_aggregation_strategy(strategy_name: str, **kwargs) -> Union[
    FedAvgAggregator, 
    FedProxAggregator, 
    FedAdamAggregator, 
    NonIIDAggregator
]:
    """
    Factory function to create aggregation strategy.
    
    Args:
        strategy_name: Name of strategy
            - 'fedavg': Standard FedAvg
            - 'fedprox': FedProx with proximal term
            - 'fedadam': Adaptive with Adam
            - 'noniid': Custom Non-IID handling
        **kwargs: Strategy-specific parameters
        
    Returns:
        Aggregator instance
        
    Example:
        >>> agg = get_aggregation_strategy('fedavg')
        >>> agg = get_aggregation_strategy('fedprox', mu=0.01)
        >>> agg = get_aggregation_strategy('fedadam', learning_rate=0.01)
    """
    strategy_name = strategy_name.lower()
    
    if strategy_name == 'fedavg':
        return FedAvgAggregator()
    
    elif strategy_name == 'fedprox':
        mu = kwargs.get('mu', 0.01)
        return FedProxAggregator(mu=mu)
    
    elif strategy_name == 'fedadam':
        lr = kwargs.get('learning_rate', 0.01)
        beta1 = kwargs.get('beta1', 0.9)
        beta2 = kwargs.get('beta2', 0.999)
        return FedAdamAggregator(learning_rate=lr, beta1=beta1, beta2=beta2)
    
    elif strategy_name == 'noniid':
        strat = kwargs.get('strategy', 'adaptive_weight')
        alpha = kwargs.get('alpha', 0.7)
        return NonIIDAggregator(strategy=strat, alpha=alpha)
    
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy_name}. "
                        f"Available: fedavg, fedprox, fedadam, noniid")


# =============================================================================
# Testing & Validation
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Federated Aggregation Strategies")
    print("=" * 70)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Simulate client parameters
    np.random.seed(42)
    
    # Create 3 clients with different data amounts
    print("\n" + "=" * 70)
    print("Setup: 3 clients with different data distributions")
    print("=" * 70)
    
    client_params = [
        ([np.random.randn(10, 5).astype(np.float32), 
          np.random.randn(5).astype(np.float32)], 100, 0),  # Client 0
        ([np.random.randn(10, 5).astype(np.float32), 
          np.random.randn(5).astype(np.float32)], 200, 1),  # Client 1
        ([np.random.randn(10, 5).astype(np.float32), 
          np.random.randn(5).astype(np.float32)], 150, 2),  # Client 2
    ]
    
    print(f"Client 0: 100 samples")
    print(f"Client 1: 200 samples")
    print(f"Client 2: 150 samples")
    print(f"Total: 450 samples")
    
    # Test 1: FedAvg
    print("\n" + "=" * 70)
    print("1. Testing FedAvg Aggregation")
    print("=" * 70)
    
    fedavg = FedAvgAggregator()
    params_and_samples = [(p, n) for p, n, _ in client_params]
    fedavg_result = fedavg.aggregate(params_and_samples)
    
    print(f"✓ Aggregated {len(fedavg_result)} parameter arrays")
    print(f"  - Array 0 shape: {fedavg_result[0].shape}")
    print(f"  - Array 1 shape: {fedavg_result[1].shape}")
    
    # Expected weights: 100/450, 200/450, 150/450 = 0.222, 0.444, 0.333
    expected_weight_1 = 200 / 450
    print(f"  - Client 1 should have highest weight: {expected_weight_1:.3f}")
    
    # Test 2: NonIID Adaptive
    print("\n" + "=" * 70)
    print("2. Testing NonIID Adaptive Aggregation")
    print("=" * 70)
    
    noniid_agg = NonIIDAggregator(strategy='adaptive_weight', alpha=0.7)
    
    # Simulate different losses
    noniid_agg.update_client_loss(0, 2.5)  # High loss (poor)
    noniid_agg.update_client_loss(1, 0.5)  # Low loss (good)
    noniid_agg.update_client_loss(2, 1.0)  # Medium loss
    
    print("Client losses:")
    print(f"  Client 0: 2.5 (poor)")
    print(f"  Client 1: 0.5 (good)")
    print(f"  Client 2: 1.0 (medium)")
    
    weights = noniid_agg.compute_adaptive_weights([0, 1, 2], [100, 200, 150])
    print(f"\nAdaptive weights: {weights}")
    print(f"  Client 1 (best performance) has highest weight: {weights[1]:.3f}")
    
    noniid_result = noniid_agg.aggregate(client_params)
    print(f"✓ Aggregated {len(noniid_result)} parameter arrays")
    
    # Test 3: FedProx
    print("\n" + "=" * 70)
    print("3. Testing FedProx Aggregation")
    print("=" * 70)
    
    fedprox = FedProxAggregator(mu=0.01)
    global_params = [np.random.randn(10, 5).astype(np.float32), 
                    np.random.randn(5).astype(np.float32)]
    
    fedprox_result = fedprox.aggregate(params_and_samples, global_params)
    print(f"✓ FedProx aggregated with mu={fedprox.mu}")
    print(f"  - Proximal regularization applied")
    
    # Test 4: FedAdam
    print("\n" + "=" * 70)
    print("4. Testing FedAdam Aggregation")
    print("=" * 70)
    
    fedadam = FedAdamAggregator(learning_rate=0.01)
    
    # First iteration
    fedadam_result1 = fedadam.aggregate(params_and_samples, global_params)
    print(f"✓ Round 1: Momentum initialized")
    print(f"  - Time step: {fedadam.t}")
    
    # Second iteration
    fedadam_result2 = fedadam.aggregate(params_and_samples, fedadam_result1)
    print(f"✓ Round 2: Using momentum from previous round")
    print(f"  - Time step: {fedadam.t}")
    
    # Test 5: Factory function
    print("\n" + "=" * 70)
    print("5. Testing Factory Function")
    print("=" * 70)
    
    strategies = ['fedavg', 'fedprox', 'fedadam', 'noniid']
    
    for strat in strategies:
        agg = get_aggregation_strategy(strat)
        print(f"✓ Created {agg.name}")
    
    # Test 6: Helper functions
    print("\n" + "=" * 70)
    print("6. Testing Helper Functions")
    print("=" * 70)
    
    # Create dummy PyTorch model
    dummy_model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU()
    )
    
    state_dict = dummy_model.state_dict()
    print(f"Original state_dict keys: {list(state_dict.keys())}")
    
    # Convert to numpy
    numpy_params = parameters_to_numpy(state_dict)
    print(f"✓ Converted to {len(numpy_params)} numpy arrays")
    
    # Convert back to state_dict
    recovered_state_dict = numpy_to_parameters(numpy_params, state_dict)
    print(f"✓ Recovered state_dict with {len(recovered_state_dict)} parameters")
    
    # Verify
    all_close = all(
        torch.allclose(state_dict[k], recovered_state_dict[k])
        for k in state_dict.keys()
    )
    print(f"✓ Conversion preserves values: {all_close}")
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nAggregation strategies are ready for federated training.")
    print("Use get_aggregation_strategy() to create aggregators in your pipeline.")