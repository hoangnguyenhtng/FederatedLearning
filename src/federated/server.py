
import flwr as fl
from flwr.common import Parameters, Scalar
from flwr.server.strategy import FedAvg
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn

class FedPerStrategy(FedAvg):

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
                 initial_parameters: Optional[Parameters] = None,
                 global_model: Optional[nn.Module] = None):  # ✅ Reference to update after agg

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

        # ✅ Reference to global model — updated after each aggregation
        # Ensures _client_fn deepcopy gets latest shared weights (personal head init improves)
        self.global_model = global_model

        # Track metrics (train_loss from fit, test_loss + accuracy from evaluate)
        self.metrics_history = {
            'train_loss': [],
            'test_loss': [],
            'accuracy': [],
            'round': []
        }

        # ✅ Early stopping to prevent overfitting
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.patience_counter = 0
        self.patience = 15  # Stop if no improvement for 15 rounds
        self.best_round = 0
        self._best_params = None  # Cache best aggregated params
        self._should_stop = False  # ✅ Flag to actually stop Flower simulation
        
    def aggregate_fit(self,
                     server_round: int,
                     results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
                     failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]]):

        if not results:
            return None, {}

        # Call parent class aggregation (FedAvg)
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # ✅ FIX: Update global_model shared params with aggregated weights
        # This ensures _client_fn's deepcopy gets the latest federated shared representation
        # → personal head in next round starts from better initialization
        if aggregated_parameters is not None and self.global_model is not None:
            try:
                param_arrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
                shared_params = self.global_model.get_shared_parameters()
                param_names = list(shared_params.keys())
                from collections import OrderedDict
                params_dict = OrderedDict()
                dev = next(self.global_model.parameters()).device
                for name, arr in zip(param_names, param_arrays):
                    params_dict[name] = torch.tensor(arr, dtype=torch.float32, device=dev)
                self.global_model.set_shared_parameters(params_dict)
            except Exception as e:
                print(f"[Server] ⚠️  Could not update global_model shared params: {e}")

        # Extract train_loss metrics from clients (weighted average)
        train_losses = []
        num_examples = []
        for _, fit_res in results:
            if fit_res.metrics and 'train_loss' in fit_res.metrics:
                train_losses.append(fit_res.metrics['train_loss'])
                num_examples.append(fit_res.num_examples)

        if train_losses:
            total_examples = sum(num_examples)
            weighted_loss = sum(loss * n for loss, n in zip(train_losses, num_examples)) / total_examples
            self.metrics_history['train_loss'].append((server_round, weighted_loss))
            self.metrics_history['round'].append(server_round)
            print(f"[Server] Round {server_round} - Avg Train Loss: {weighted_loss:.4f}")

        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self,
                          server_round: int,
                          results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
                          failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]]):

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
        weighted_test_loss = None
        weighted_accuracy = None
        
        if test_losses:
            weighted_test_loss = sum(loss * n for loss, n in zip(test_losses, num_examples)) / total_examples
            self.metrics_history['test_loss'].append((server_round, weighted_test_loss))

        if accuracies:
            weighted_accuracy = sum(acc * n for acc, n in zip(accuracies, num_examples)) / total_examples
            self.metrics_history['accuracy'].append((server_round, weighted_accuracy))
            _loss_str = f"{weighted_test_loss:.4f}" if weighted_test_loss is not None else "N/A"
            print(f"[Server] Round {server_round} - Avg Test Loss: {_loss_str}, Avg Accuracy: {weighted_accuracy:.4f}")

        # Return metrics to Flower History (for accuracy curve in plot)
        out_metrics = dict(aggregated_metrics) if aggregated_metrics else {}
        if weighted_test_loss is not None:
            out_metrics['test_loss'] = float(weighted_test_loss)
        if weighted_accuracy is not None:
            out_metrics['accuracy'] = float(weighted_accuracy)

        # Flower only records history when aggregated_loss is not None
        if aggregated_loss is None and results:
            losses = [float(evaluate_res.loss) for _, evaluate_res in results if evaluate_res.loss is not None]
            if losses:
                aggregated_loss = float(sum(losses) / len(losses))

        # ✅ Early stopping check
        check_loss = weighted_test_loss if weighted_test_loss is not None else aggregated_loss
        if check_loss is not None:
            if check_loss < self.best_loss - 0.001:
                self.best_loss = check_loss
                self.best_accuracy = weighted_accuracy or 0.0
                self.patience_counter = 0
                self.best_round = server_round
                # Cache best parameters from global model
                if self.global_model is not None:
                    self._best_params = {
                        k: v.detach().cpu().clone()
                        for k, v in self.global_model.state_dict().items()
                    }
                print(f"[Server] ⭐ New best model at round {server_round}: "
                      f"loss={check_loss:.4f}, accuracy={self.best_accuracy:.4f}")
            else:
                self.patience_counter += 1
                print(f"[Server] ⏳ No improvement for {self.patience_counter}/{self.patience} rounds "
                      f"(best: round {self.best_round}, loss={self.best_loss:.4f}, acc={self.best_accuracy:.4f})")

            if self.patience_counter >= self.patience:
                print(f"\n[Server] 🛑 EARLY STOPPING at round {server_round}!")
                print(f"[Server] Best model was at round {self.best_round} "
                      f"with loss={self.best_loss:.4f}, accuracy={self.best_accuracy:.4f}")
                # Restore best model parameters
                if self._best_params is not None and self.global_model is not None:
                    self.global_model.load_state_dict(self._best_params)
                    print(f"[Server] ✅ Restored best model parameters from round {self.best_round}")
                self._should_stop = True  # ✅ Signal Flower to stop

        return aggregated_loss, out_metrics

    def configure_fit(self, server_round, parameters, client_manager):
        """Override to stop training when early stopping triggers."""
        if self._should_stop:
            print(f"[Server] ⏹️  Skipping fit round {server_round} (early stopped)")
            return []  # Empty list → Flower skips this round
        return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Override to stop evaluation when early stopping triggers."""
        if self._should_stop:
            print(f"[Server] ⏹️  Skipping evaluate round {server_round} (early stopped)")
            return []  # Empty list → Flower skips this round
        return super().configure_evaluate(server_round, parameters, client_manager)


def get_initial_parameters(model: nn.Module) -> Parameters:

    shared_params = model.get_shared_parameters()
    
    # Convert to numpy arrays
    param_arrays = [param.cpu().detach().numpy() for param in shared_params.values()]
    
    # Convert to Flower Parameters
    return fl.common.ndarrays_to_parameters(param_arrays)


def _normalize_fl_root_config(config: Dict) -> Tuple[Dict, Dict]:

    if config is not None and "federated" in config:
        return config["federated"], config
    return config, {"federated": config, "training": {}}


def get_on_fit_config_fn(full_config: Dict):

    train = full_config.get("training", {}) or {}
    fed = full_config.get("federated", {}) or {}

    def fit_config(server_round: int):
        """Config cho training round"""
        return {
            "server_round": server_round,
            "local_epochs": train.get("local_epochs", fed.get("local_epochs", 3)),
            "batch_size": train.get("batch_size", fed.get("batch_size", 32)),
            "learning_rate": train.get("learning_rate", fed.get("learning_rate", 0.001)),
        }

    return fit_config


def get_on_evaluate_config_fn(full_config: Dict):

    train = full_config.get("training", {}) or {}

    def evaluate_config(server_round: int):
        """Config cho evaluation round"""
        return {
            "server_round": server_round,
            "batch_size": train.get("batch_size", 32),
            "personalize_epochs": train.get("personalize_epochs_eval", 2),
        }

    return evaluate_config


def create_fedper_strategy(model: nn.Module, config: Dict) -> FedPerStrategy:

    initial_parameters = get_initial_parameters(model)
    fed_cfg, full_root = _normalize_fl_root_config(config)

    strategy = FedPerStrategy(
        fraction_fit=fed_cfg.get("fraction_fit", 0.6),
        fraction_evaluate=fed_cfg.get("fraction_evaluate", 0.5),
        min_fit_clients=fed_cfg.get("min_fit_clients", 3),
        min_evaluate_clients=fed_cfg.get("min_evaluate_clients", 2),
        min_available_clients=fed_cfg.get("min_available_clients", 5),
        initial_parameters=initial_parameters,
        global_model=model,
        on_fit_config_fn=get_on_fit_config_fn(full_root),
        on_evaluate_config_fn=get_on_evaluate_config_fn(full_root),
    )

    return strategy


def start_server(model: nn.Module,
                config: Dict,
                num_rounds: int = 50,
                server_address: str = "0.0.0.0:8080"):

    # Create strategy
    strategy = create_fedper_strategy(model, config)
    fed_cfg, _ = _normalize_fl_root_config(config)

    print(f"Starting Federated Learning Server...")
    print(f"Server address: {server_address}")
    print(f"Number of rounds: {num_rounds}")
    print(f"Fraction fit: {fed_cfg.get('fraction_fit', 0.6)}")
    print(f"Min fit clients: {fed_cfg.get('min_fit_clients', 5)}")
    
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
        num_classes=5  # Rating prediction: 5 classes (1-5 → 0-4)
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