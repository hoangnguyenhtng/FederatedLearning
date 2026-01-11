"""
Federated Server Implementation

Nhiệm vụ:
1. Khởi tạo global model
2. Chọn clients tham gia mỗi round
3. Aggregate parameters từ clients (FedAvg)
4. Distribute global model về clients
5. Track metrics và convergence
"""
import flwr as fl
from flwr.common import Parameters, Scalar
from flwr.server.strategy import FedAvg
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn

class FedPerStrategy(FedAvg):
    """
    Custom Federated Learning Strategy cho FedPer
    
    Khác biệt với FedAvg standard:
    - Chỉ aggregate SHARED parameters
    - Personal parameters không được aggregate
    - Có thể add custom logic cho Non-IID data
    """
    
    def __init__(self,
                 fraction_fit: float = 0.6,
                 fraction_evaluate: float = 0.5,
                 min_fit_clients: int = 5,
                 min_evaluate_clients: int = 3,
                 min_available_clients: int = 8,
                 evaluate_fn=None,
                 on_fit_config_fn=None,
                 on_evaluate_config_fn=None,
                 accept_failures: bool = True,
                 initial_parameters: Optional[Parameters] = None):
        """
        Args:
            fraction_fit: Tỷ lệ clients tham gia training mỗi round
            fraction_evaluate: Tỷ lệ clients tham gia evaluation
            min_fit_clients: Số clients tối thiểu cho training
            min_evaluate_clients: Số clients tối thiểu cho evaluation
            min_available_clients: Số clients tối thiểu available
            evaluate_fn: Function để evaluate global model
            on_fit_config_fn: Config function cho training
            on_evaluate_config_fn: Config function cho evaluation
            accept_failures: Có chấp nhận failures không
            initial_parameters: Initial model parameters
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters
        )
        
        # Track metrics
        self.metrics_history = {
            'train_loss': [],
            'test_loss': [],
            'accuracy': [],
            'round': []
        }
        
    def aggregate_fit(self,
                     server_round: int,
                     results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
                     failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]]):
        """
        Aggregate parameters từ clients sau training
        
        Process:
        1. Nhận shared parameters từ các clients
        2. Weighted average theo số lượng samples
        3. Return aggregated parameters
        
        Args:
            server_round: Số round hiện tại
            results: List (client_proxy, fit_result) từ các clients
            failures: List các clients failed
            
        Returns:
            aggregated_parameters: Aggregated model parameters
            metrics: Aggregated metrics
        """
        if not results:
            return None, {}
        
        # Call parent class aggregation (FedAvg)
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Extract metrics từ clients
        train_losses = []
        num_examples = []
        
        for _, fit_res in results:
            if fit_res.metrics and 'train_loss' in fit_res.metrics:
                train_losses.append(fit_res.metrics['train_loss'])
                num_examples.append(fit_res.num_examples)
        
        # Weighted average của train loss
        if train_losses:
            total_examples = sum(num_examples)
            weighted_loss = sum(loss * n for loss, n in zip(train_losses, num_examples)) / total_examples
            
            self.metrics_history['train_loss'].append(weighted_loss)
            self.metrics_history['round'].append(server_round)
            
            print(f"[Server] Round {server_round} - Avg Train Loss: {weighted_loss:.4f}")
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self,
                          server_round: int,
                          results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
                          failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]]):
        """
        Aggregate evaluation results từ clients
        
        Args:
            server_round: Số round hiện tại
            results: List (client_proxy, evaluate_result)
            failures: List các clients failed
            
        Returns:
            aggregated_loss: Aggregated loss
            aggregated_metrics: Aggregated metrics
        """
        if not results:
            return None, {}
        
        # Call parent class aggregation
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        # Extract metrics
        test_losses = []
        accuracies = []
        num_examples = []
        
        for _, evaluate_res in results:
            if evaluate_res.metrics:
                if 'test_loss' in evaluate_res.metrics:
                    test_losses.append(evaluate_res.metrics['test_loss'])
                if 'accuracy' in evaluate_res.metrics:
                    accuracies.append(evaluate_res.metrics['accuracy'])
                num_examples.append(evaluate_res.num_examples)
        
        # Weighted averages
        total_examples = sum(num_examples)
        
        if test_losses:
            weighted_test_loss = sum(loss * n for loss, n in zip(test_losses, num_examples)) / total_examples
            self.metrics_history['test_loss'].append(weighted_test_loss)
        
        if accuracies:
            weighted_accuracy = sum(acc * n for acc, n in zip(accuracies, num_examples)) / total_examples
            self.metrics_history['accuracy'].append(weighted_accuracy)
            
            print(f"[Server] Round {server_round} - Avg Test Loss: {weighted_test_loss:.4f}, Avg Accuracy: {weighted_accuracy:.4f}")
        
        return aggregated_loss, aggregated_metrics


def get_initial_parameters(model: nn.Module) -> Parameters:
    """
    Get initial parameters từ model để khởi tạo server
    
    Args:
        model: PyTorch model
        
    Returns:
        Parameters object cho Flower
    """
    shared_params = model.get_shared_parameters()
    
    # Convert to numpy arrays
    param_arrays = [param.cpu().detach().numpy() for param in shared_params.values()]
    
    # Convert to Flower Parameters
    return fl.common.ndarrays_to_parameters(param_arrays)


def get_on_fit_config_fn(config: Dict):
    """
    Tạo config function cho training
    
    Returns:
        Function that returns config dict cho mỗi training round
    """
    def fit_config(server_round: int):
        """Config cho training round"""
        return {
            "server_round": server_round,
            "local_epochs": config.get('local_epochs', 3),
            "batch_size": config.get('batch_size', 32),
            "learning_rate": config.get('learning_rate', 0.001)
        }
    
    return fit_config


def get_on_evaluate_config_fn(config: Dict):
    """
    Tạo config function cho evaluation
    
    Returns:
        Function that returns config dict cho mỗi evaluation round
    """
    def evaluate_config(server_round: int):
        """Config cho evaluation round"""
        return {
            "server_round": server_round,
            "batch_size": config.get('batch_size', 32)
        }
    
    return evaluate_config


def create_fedper_strategy(model: nn.Module, config: Dict) -> FedPerStrategy:
    """
    Factory function để tạo FedPer strategy
    
    Args:
        model: Initial model
        config: Configuration dict
        
    Returns:
        FedPerStrategy instance
    """
    initial_parameters = get_initial_parameters(model)
    
    strategy = FedPerStrategy(
        fraction_fit=config.get('fraction_fit', 0.6),
        fraction_evaluate=config.get('fraction_evaluate', 0.5),
        min_fit_clients=config.get('min_fit_clients', 3),
        min_evaluate_clients=config.get('min_evaluate_clients', 2),
        min_available_clients=config.get('min_available_clients', 5),
        initial_parameters=initial_parameters,
        on_fit_config_fn=get_on_fit_config_fn(config),
        on_evaluate_config_fn=get_on_evaluate_config_fn(config)
    )
    
    return strategy


def start_server(model: nn.Module,
                config: Dict,
                num_rounds: int = 50,
                server_address: str = "0.0.0.0:8080"):
    """
    Start Federated Learning Server
    
    Args:
        model: Initial model architecture
        config: Configuration dict
        num_rounds: Số rounds training
        server_address: Server address
    """
    # Create strategy
    strategy = create_fedper_strategy(model, config)
    
    print(f"Starting Federated Learning Server...")
    print(f"Server address: {server_address}")
    print(f"Number of rounds: {num_rounds}")
    print(f"Fraction fit: {config.get('fraction_fit', 0.6)}")
    print(f"Min fit clients: {config.get('min_fit_clients', 5)}")
    
    # Start server
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )
    
    print("Training completed!")
    
    # Return metrics history
    return strategy.metrics_history


# Example usage
if __name__ == "__main__":
    from src.models.multimodal_encoder import MultiModalEncoder
    from src.models.recommendation_model import FedPerRecommender
    
    # Initialize model
    multimodal_encoder = MultiModalEncoder()
    model = FedPerRecommender(
        multimodal_encoder=multimodal_encoder,
        num_items=1000
    )
    
    # Configuration
    config = {
        'fraction_fit': 0.6,
        'fraction_evaluate': 0.5,
        'min_fit_clients': 3,
        'min_evaluate_clients': 2,
        'min_available_clients': 5,
        'local_epochs': 3,
        'batch_size': 32,
        'learning_rate': 0.001
    }
    
    # Start server (simulation mode)
    print("Server configuration loaded. Ready to start training.")
    print(f"Config: {config}")