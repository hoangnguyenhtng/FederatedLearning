"""
Data Generation Module
Handles synthetic data generation and Non-IID data splitting
"""

from .synthetic_data_generator import SyntheticDataGenerator
from .non_iid_data_splitter import NonIIDDataSplitter
from .federated_dataloader import FederatedDataLoader

__all__ = [
    'SyntheticDataGenerator',
    'NonIIDDataSplitter', 
    'FederatedDataLoader'
]