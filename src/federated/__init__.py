"""
Federated Learning Package

Components:
- Client: Federated client implementation  
- Server: Federated server with FedPer strategy
- Aggregator: Model aggregation strategies
- Privacy: Differential privacy utilities
"""

from .client import create_client_fn, FedPerClient
from .server import (
    FedPerStrategy,
    get_initial_parameters,
    get_on_fit_config_fn,
    get_on_evaluate_config_fn,
    start_server
)
from .aggregator import (
    FedAvgAggregator,
    FedProxAggregator,
    FedAdamAggregator,
    NonIIDAggregator,
    get_aggregation_strategy,
    parameters_to_numpy,
    numpy_to_parameters
)
from .privacy import (
    apply_differential_privacy,
    compute_privacy_budget,
    get_dp_optimizer,
    clip_gradients,
    add_noise_to_gradients
)

__all__ = [
    # Client
    'FedPerClient',
    'create_client_fn',
    
    # Server
    'FedPerStrategy',
    'get_initial_parameters',
    'get_on_fit_config_fn',
    'get_on_evaluate_config_fn',
    'start_server',
    
    # Aggregator
    'FedAvgAggregator',
    'FedProxAggregator', 
    'FedAdamAggregator',
    'NonIIDAggregator',
    'get_aggregation_strategy',
    'parameters_to_numpy',
    'numpy_to_parameters',
    
    # Privacy
    'apply_differential_privacy',
    'compute_privacy_budget',
    'get_dp_optimizer',
    'clip_gradients',
    'add_noise_to_gradients'
]