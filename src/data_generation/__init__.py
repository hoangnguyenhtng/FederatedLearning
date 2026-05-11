"""
Data Generation Module
Handles data generation, processing, and federated data loading
"""

# Use lazy imports to avoid breaking when optional modules are missing
__all__ = []

try:
    from .federated_dataloader import get_federated_dataloaders
    __all__.append('get_federated_dataloaders')
except ImportError:
    pass

try:
    from .amazon_dataloader import get_amazon_dataloaders, AmazonDataset
    __all__.extend(['get_amazon_dataloaders', 'AmazonDataset'])
except ImportError:
    pass

try:
    from .generate_demo_data import generate_demo_data
    __all__.append('generate_demo_data')
except ImportError:
    pass