"""
Federated Training Pipeline - CORRECT VERSION
Matched to your FedPerRecommender signature
"""

import sys
import os
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
import argparse
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Local imports
# Note: model_factory is available but we create model directly for clarity
# from src.models.model_factory import create_model

from src.models.multimodal_encoder import MultiModalEncoder
from src.models.recommendation_model import FedPerRecommender
from src.training.local_trainer import LocalTrainer
from src.federated.server import create_fedper_strategy, FedPerStrategy
from src.federated.client import FedPerClient, create_client_fn
from src.federated.privacy import apply_differential_privacy, compute_privacy_budget  
from src.data_generation.federated_dataloader import get_federated_dataloaders
from src.training.training_utils import (
    calculate_metrics,
    save_checkpoint,
    load_checkpoint,
    setup_logging,
    resolve_amazon_federated_data_dir,
    experiments_base_dir,
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _metric_series_from_flwr_history(
    metrics_distributed: Optional[Dict],
    metric_name: str,
) -> Tuple[List[int], List[float]]:
    """
    Flower History.metrics_distributed: Dict[str, List[Tuple[int, Scalar]]],
    ví dụ {'accuracy': [(1, 0.85), (2, 0.88), ...]}. Sau json.load, tuple thành list.

    Định dạng cũ (sai): round -> {'accuracy': ...} — vẫn hỗ trợ nếu file JSON lưu theo kiểu đó.
    """
    if not metrics_distributed:
        return [], []

    if metric_name in metrics_distributed:
        series = metrics_distributed[metric_name]
        rounds: List[int] = []
        vals: List[float] = []
        if isinstance(series, list):
            for item in series:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    try:
                        rounds.append(int(item[0]))
                        vals.append(float(item[1]))
                    except (TypeError, ValueError):
                        continue
        if rounds:
            return rounds, vals

    sample = metrics_distributed.get(next(iter(metrics_distributed)))
    if isinstance(sample, dict) and metric_name in sample:
        rounds = []
        vals = []
        def _rk(k):
            try:
                return int(k)
            except (TypeError, ValueError):
                return 0
        for k in sorted(metrics_distributed.keys(), key=_rk):
            m = metrics_distributed[k]
            if isinstance(m, dict) and m.get(metric_name) is not None:
                try:
                    rounds.append(int(k))
                    vals.append(float(m[metric_name]))
                except (TypeError, ValueError):
                    continue
        return rounds, vals

    return [], []


class FederatedTrainingPipeline:
    """Complete federated training pipeline"""
    
    def __init__(self, config: dict, experiment_dir: Path):
        """
        Initialize pipeline
        
        Args:
            config: Configuration dictionary
            experiment_dir: Directory to save experiments
        """
        self.config = config
        self.experiment_dir = experiment_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create subdirectories
        self.models_dir = experiment_dir / "models"
        self.logs_dir = experiment_dir / "logs"
        self.metrics_dir = experiment_dir / "metrics"
        
        for dir_path in [self.models_dir, self.logs_dir, self.metrics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.global_model = self._create_model()
        self.dataloaders = self._load_data()
        
        logger.info(f"🖥️  Using device: {self.device}")
        logger.info(f"📊 Number of clients: {len(self.dataloaders)}")
    
    def _create_model(self) -> nn.Module:
        """
        Create global model - CORRECT VERSION
        Your model needs: multimodal_encoder object, NOT dimensions!
        """
        logger.info("🔨 Creating global model...")
        
        model_config = self.config['model']
        
        # Step 1: Create MultiModalEncoder first
        logger.info("   Creating MultiModalEncoder...")
        multimodal_encoder = MultiModalEncoder(
            text_dim=model_config.get('text_embedding_dim', 384),
            image_dim=model_config.get('image_embedding_dim', 2048),
            behavior_dim=model_config.get(
                'behavior_embedding_dim',
                model_config.get('behavior_dim', 32),
            ),
            hidden_dim=model_config.get(
                'hidden_dim',
                model_config.get('multimodal_hidden_dim', 256),
            ),
            output_dim=model_config.get('multimodal_output_dim', 384),
        )
        
        # Step 2: Create FedPerRecommender with encoder
        logger.info("   Creating FedPerRecommender...")
        
        # Get layer configurations
        shared_dims = model_config.get('shared_hidden_dims', [512, 256, 128])
        personal_dims = model_config.get('personal_hidden_dims', [64, 32])
        # Rating prediction: 5 classes (ratings 1-5, mapped to 0-4)
        num_classes = model_config.get('num_classes', 5)  # Changed from 10000 to 5
        dropout = model_config.get('dropout', 0.2)

        model = FedPerRecommender(
            multimodal_encoder=multimodal_encoder,  # ✅ Pass encoder object
            shared_hidden_dims=shared_dims,
            personal_hidden_dims=personal_dims,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        shared_params = sum(p.numel() for p in model.get_shared_parameters().values())
        personal_params = sum(p.numel() for p in model.get_personal_parameters().values())
        
        logger.info(f"✅ Model created successfully")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Shared parameters: {shared_params:,} ({shared_params/total_params*100:.1f}%)")
        logger.info(f"   Personal parameters: {personal_params:,} ({personal_params/total_params*100:.1f}%)")
        
        return model
    
    def _load_data(self) -> Dict[int, Tuple[DataLoader, DataLoader]]:
        """Load federated dataloaders - Auto-detect Amazon or Synthetic data"""
        logger.info("📂 Loading federated data...")
        
        num_clients = self.config['federated']['num_clients']
        batch_size = self.config['training']['batch_size']
        test_split = self.config['training'].get('test_split', 0.2)
        
        amazon_dir = resolve_amazon_federated_data_dir(self.config, cwd=project_root)
        paths_cfg = self.config.get('paths') or {}
        synthetic_dir = (project_root / paths_cfg.get('data_dir', 'data') / 'simulated_clients').resolve()

        if amazon_dir is not None:
            logger.info(f"🎉 Using AMAZON / processed pickles: {amazon_dir}")
            from src.data_generation.amazon_dataloader import get_amazon_dataloaders

            dataloaders = get_amazon_dataloaders(
                num_clients=num_clients,
                data_dir=str(amazon_dir),
                batch_size=batch_size,
                test_split=test_split,
            )

        elif synthetic_dir.exists():
            # Fallback to synthetic data
            logger.warning("⚠️  Using SYNTHETIC data (contains random noise!)")
            logger.warning("   For better results, use Amazon data:")
            logger.warning("   1. PowerShell -ExecutionPolicy Bypass -File download_amazon_data.ps1")
            logger.warning("   2. python src\\data_generation\\process_amazon_data.py")
            
            dataloaders_list = get_federated_dataloaders(
                num_clients=num_clients,
                data_dir=synthetic_dir,
                batch_size=batch_size,
                test_split=test_split
            )
            
            # Convert list -> dict
            dataloaders = {}
            for client_id, loaders in enumerate(dataloaders_list):
                if loaders is not None and len(loaders) == 2:
                    train_loader, test_loader = loaders
                    if train_loader is not None and test_loader is not None:
                        dataloaders[client_id] = (train_loader, test_loader)
        else:
            raise FileNotFoundError(
                f"❌ No data found!\n"
                f"Checked processed Amazon (paths.data_processed + legacy data/amazon_2023_processed), "
                f"then synthetic:\n"
                f"  - Synthetic: {synthetic_dir}\n\n"
                f"Multi-category preprocess:\n"
                f"  python src/data_generation/process_amazon_multi_category.py --config configs/config_multi_category.yaml\n\n"
                f"Hoặc single-path Amazon + synthetic fallback:\n"
                f"  PowerShell ... download_amazon_data.ps1\n"
                f"  python src/data_generation/process_amazon_data.py\n\n"
                f"  python src/data_generation/main_data_generation.py"
            )
        
        if not dataloaders:
            raise ValueError(f"No dataloaders loaded. Expected {num_clients} clients")
        
        logger.info(f"✅ Loaded {len(dataloaders)} client dataloaders (expected {num_clients})")
        
        # Warn if some clients are missing
        if len(dataloaders) < num_clients:
            missing = set(range(num_clients)) - set(dataloaders.keys())
            logger.warning(f"⚠️  Missing dataloaders for clients: {missing}")
        
        # Log first client stats
        first_client_id = list(dataloaders.keys())[0]
        train_loader, val_loader = dataloaders[first_client_id]
        logger.info(f"   Client {first_client_id}:")
        logger.info(f"     Train batches: {len(train_loader)}")
        logger.info(f"     Val batches: {len(val_loader)}")
        
        return dataloaders
    
    def _client_fn(self, context):
        """
        Create a client function for Flower simulation
        
        Args:
            context: Flower Context object or client ID (string/int) for backward compatibility
            
        Returns:
            FedPerClient instance
        """
        # Extract client ID from context
        # In Flower 1.7.0, client_fn receives client ID as string/int
        cid = None
        
        # Try to extract client ID from context (string/int/dict)
        try:
            if isinstance(context, (str, int)):
                cid = int(context)
            elif isinstance(context, dict):
                cid = int(context.get('cid', context.get('partition-id', 0)))
            elif hasattr(context, 'cid'):
                cid = int(context.cid)
            elif hasattr(context, 'partition_id'):
                cid = int(context.partition_id)
            else:
                # Last resort: try to convert
                cid = int(str(context))
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Failed to extract client ID from context: {context}, type: {type(context)}, error: {e}")
            raise ValueError(f"Cannot extract client ID from context: {context}")
        
        # Ensure cid is integer
        cid = int(cid)
        
        # Validate client ID exists in dataloaders
        if cid not in self.dataloaders:
            available = list(self.dataloaders.keys())
            logger.error(f"Client {cid} not found in dataloaders. Available clients: {available}")
            raise ValueError(
                f"Client {cid} not found in dataloaders. "
                f"Available clients: {available}. "
                f"Total clients configured: {len(available)}"
            )
        
        # Get client data
        train_loader, test_loader = self.dataloaders[cid]
        
        # Validate loaders
        if train_loader is None or test_loader is None:
            raise ValueError(f"Client {cid} has None loaders")
        
        # Create a copy of the model for this client
        import copy
        client_model = copy.deepcopy(self.global_model)
        client_model.to(self.device)
        
        # Create client
        numpy_client = FedPerClient(
            client_id=cid,
            model=client_model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=str(self.device),
            local_epochs=self.config['training']['local_epochs'],
            learning_rate=self.config['training']['learning_rate'],
            apply_dp=self.config.get('privacy', {}).get('differential_privacy', False)
        )
        
        # Convert NumPyClient to Client (required by newer Flower versions)
        try:
            client = numpy_client.to_client()
        except AttributeError:
            # If to_client() doesn't exist, return NumPyClient directly
            # (older Flower versions)
            client = numpy_client
        
        logger.debug(f"✅ Created client {cid}")
        return client


    
    def train(self):
        """Run federated training"""
        
        logger.info("\n" + "="*70)
        logger.info("🚀 STARTING FEDERATED TRAINING")
        logger.info("="*70)
        
        # Create strategy
        strategy = create_fedper_strategy(
            model=self.global_model,
            config=self.config,
        )

        # ----------------------------------------------------------------
        # FIX: Tránh MemoryError khi Ray serialize closure
        # Nếu pass self._client_fn trực tiếp, Ray sẽ pickle toàn bộ `self`
        # (bao gồm 40 dataloaders + model = hàng trăm MB).
        # Thay thế: tạo closure nhẹ chỉ capture dữ liệu tối thiểu.
        # ----------------------------------------------------------------
        import copy

        # Snapshot những gì client_fn cần (không phải toàn bộ self)
        _dataloaders   = self.dataloaders          # dict {cid: (train, test)}
        _global_model  = self.global_model         # reference (not deepcopy)
        _device        = self.device
        _config        = self.config
        _logger        = logger

        def _lightweight_client_fn(context):
            """Lightweight client factory - chỉ capture data cần thiết."""
            # Extract cid
            try:
                if isinstance(context, (str, int)):
                    cid = int(context)
                elif isinstance(context, dict):
                    cid = int(context.get('cid', context.get('partition-id', 0)))
                elif hasattr(context, 'cid'):
                    cid = int(context.cid)
                elif hasattr(context, 'partition_id'):
                    cid = int(context.partition_id)
                else:
                    cid = int(str(context))
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"Cannot extract client ID from context: {context}") from exc

            if cid not in _dataloaders:
                available = list(_dataloaders.keys())
                raise ValueError(
                    f"Client {cid} not found. Available: {available}"
                )

            train_loader, test_loader = _dataloaders[cid]
            if train_loader is None or test_loader is None:
                raise ValueError(f"Client {cid} has None loaders")

            client_model = copy.deepcopy(_global_model)
            client_model.to(_device)

            numpy_client = FedPerClient(
                client_id=cid,
                model=client_model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=str(_device),
                local_epochs=_config['training']['local_epochs'],
                learning_rate=_config['training']['learning_rate'],
                apply_dp=_config.get('privacy', {}).get('differential_privacy', False),
            )

            try:
                return numpy_client.to_client()
            except AttributeError:
                return numpy_client

        client_fn = _lightweight_client_fn

        
        # Configure simulation
        num_rounds = self.config['federated']['num_rounds']
        
        logger.info(f"Configuration:")
        logger.info(f"  Rounds: {num_rounds}")
        logger.info(f"  Clients per round: {int(self.config['federated']['fraction_fit'] * self.config['federated']['num_clients'])}")
        logger.info(f"  Local epochs: {self.config['training']['local_epochs']}")
        logger.info(f"  Batch size: {self.config['training']['batch_size']}")
        logger.info(f"  Learning rate: {self.config['training']['learning_rate']}")
        logger.info("")
        
        # Run simulation
        try:
            # Reduce resource usage to avoid Ray crashes on Windows
            # Use fewer concurrent clients and lower memory per client
            # NOTE: Don't set memory limit - let Ray manage it automatically
            # Setting memory too high causes "No available node types" error
            client_resources = {
                "num_cpus": 1,
                "num_gpus": 0.0 if self.device.type == 'cpu' else 0.2
                # Removed "memory" constraint - let Ray manage memory automatically
                # This avoids "No available node types" error
            }
            
            # Set Ray environment variables before initialization to reduce memory pressure
            # This helps avoid Windows access violation errors
            os.environ.setdefault("RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE", "1")
            os.environ.setdefault("RAY_DEDUP_LOGS", "1")
            # Tat memory monitor cua Ray de tranh false positive tren Windows
            os.environ["RAY_memory_monitor_refresh_ms"] = "0"
            
            # ----------------------------------------------------------------
            # Tinh toan object_store_memory an toan cho Ray
            # Ray validation: total_ram - object_store - redis_overhead >= -3% total
            # Dung gia tri co dinh nho de dam bao vuot qua validation.
            # 512MB: du lon de serialize model weights (4.8MB) va tensors,
            # nhung nho de khong anh huong RAM cho tasks.
            # ----------------------------------------------------------------
            import psutil
            total_ram     = psutil.virtual_memory().total
            available_ram = psutil.virtual_memory().available

            # Object store: 10% of available RAM, min 256MB, max 512MB
            obj_store_mb  = max(256, min(512, int(available_ram * 0.10 / (1024**2))))
            obj_store_bytes = obj_store_mb * 1024 * 1024

            # _memory: set explicitly to bypass Ray validation check.
            # Without this, Ray computes: memory = avail - object_store.
            # If avail is low (machine busy), validation fails with ValueError.
            # By providing _memory directly, Ray skips the validation branch.
            # Use 4GB or 60% of total_ram, whichever is smaller.
            explicit_memory_bytes = min(
                4 * 1024 * 1024 * 1024,           # 4 GB hard cap
                int(total_ram * 0.60),              # 60% of physical RAM
            )

            logger.info(f"RAM total: {total_ram/1024**3:.1f}GB, Available: {available_ram/1024**3:.1f}GB")
            logger.info(f"Ray object_store: {obj_store_mb}MB, task memory: {explicit_memory_bytes/1024**3:.1f}GB")

            # Configure Ray - memory param NOT supported in this version
            ray_init_args = {
                "ignore_reinit_error": True,
                "include_dashboard": False,
                "object_store_memory": obj_store_bytes,
                "_memory": explicit_memory_bytes,   # Bypass avail_memory validation
                "num_cpus": min(4, os.cpu_count() or 2),
            }
            
            history = fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=self.config['federated']['num_clients'],
                config=fl.server.ServerConfig(num_rounds=num_rounds),
                strategy=strategy,
                client_resources=client_resources,
                ray_init_args=ray_init_args  # Pass Ray init args to reduce memory
            )
            
            logger.info("\n" + "="*70)
            logger.info("✅ TRAINING COMPLETED SUCCESSFULLY")
            logger.info("="*70)
            
            # Save results
            self._save_results(history)
            
            return history
            
        except Exception as e:
            logger.error(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _save_results(self, history):
        """Save training results"""
        
        logger.info("\n📊 Saving results...")
        
        # Save final model
        final_model_path = self.models_dir / "global_model_final.pt"
        # Create a dummy optimizer for checkpoint saving
        import torch.optim as optim
        dummy_optimizer = optim.Adam(self.global_model.parameters(), lr=0.001)
        save_checkpoint(
            model=self.global_model,
            optimizer=dummy_optimizer,
            epoch=self.config['federated']['num_rounds'],
            metrics={'status': 'completed'},
            path=str(final_model_path)
        )
        logger.info(f"✅ Saved final model: {final_model_path}")
        
        # Save history
        history_path = self.metrics_dir / "training_history.json"
        
        # Convert history to serializable format
        history_dict = {
            'losses_distributed': [(round, loss) for round, loss in history.losses_distributed],
            'losses_centralized': [(round, loss) for round, loss in history.losses_centralized] if history.losses_centralized else [],
            'metrics_distributed': history.metrics_distributed,
            'metrics_centralized': history.metrics_centralized if history.metrics_centralized else {}
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        logger.info(f"✅ Saved training history: {history_path}")
        
        # Plot results
        try:
            plot_path = self.metrics_dir / "training_curves.png"
            self._plot_training_history(history, save_path=plot_path)
            logger.info(f"✅ Saved training curves: {plot_path}")
        except Exception as e:
            logger.warning(f"⚠️  Could not plot training curves: {e}")
        
        # Save config
        config_path = self.experiment_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        logger.info(f"✅ Saved configuration: {config_path}")
    
    def _plot_training_history(self, history, save_path: Path):
        """Plot training history: Loss và Accuracy theo round"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # === Subplot 1: Loss ===
            if history.losses_distributed:
                rounds_loss, losses = zip(*history.losses_distributed)
                axes[0].plot(rounds_loss, losses, 'b-o', markersize=4, label='Distributed Loss')
            
            if history.losses_centralized:
                rounds_c, losses_c = zip(*history.losses_centralized)
                axes[0].plot(rounds_c, losses_c, 'r--', label='Centralized Loss')
            
            axes[0].set_xlabel('Round')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Loss')
            axes[0].legend()
            axes[0].grid(True)
            
            # === Subplot 2: Accuracy (Flower: metrics_distributed['accuracy'] = [(round, val), ...]) ===
            rounds_acc, vals_acc = _metric_series_from_flwr_history(
                history.metrics_distributed,
                'accuracy',
            )
            if rounds_acc and vals_acc:
                axes[1].plot(rounds_acc, vals_acc, 'g-o', markersize=4, label='Accuracy')
                axes[1].set_ylabel('Accuracy')
                axes[1].legend()
            elif history.metrics_distributed:
                axes[1].text(
                    0.5, 0.5,
                    'No accuracy metrics\n(evaluation may not have run)',
                    ha='center', va='center', transform=axes[1].transAxes, fontsize=10,
                )
            else:
                axes[1].text(
                    0.5, 0.5, 'No distributed metrics',
                    ha='center', va='center', transform=axes[1].transAxes, fontsize=10,
                )

            axes[1].set_xlabel('Round')
            axes[1].set_title('Validation Accuracy')
            axes[1].grid(True)
            axes[1].set_ylim(0, 1.02)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info("✅ Plotted Loss and Accuracy curves")
            
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")
        except Exception as e:
            logger.warning(f"Could not create plot: {e}")


def main(config_path: Optional[str] = None):
    """Main entry point."""
    # Fix Unicode encoding on Windows console
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass  # Older Python or non-reconfigurable stream
    cfg_arg = config_path
    if cfg_arg is None:
        parser = argparse.ArgumentParser(description="Federated training (FedPer)")
        parser.add_argument(
            "--config",
            type=str,
            default="configs/config.yaml",
            help="YAML (configs/config.yaml, configs/config_multi_category.yaml, …)",
        )
        cfg_arg = parser.parse_args().config

    resolved = Path(cfg_arg)
    if not resolved.is_absolute():
        resolved = (project_root / resolved).resolve()

    if not resolved.exists():
        logger.error(f"❌ Config file not found: {resolved}")
        return

    with open(resolved, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    exp_block = config.get("experiment") or {}
    thesis_block = config.get("thesis") or {}
    exp_name = (
        exp_block.get("name")
        or thesis_block.get("experiment_name")
        or "fedper_experiment"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = experiments_base_dir(config, cwd=project_root)
    exp_dir = base / f"{exp_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Print header
    print("="*70)
    print("FEDERATED MULTI-MODAL RECOMMENDATION SYSTEM")
    print("Training Pipeline with FedPer Architecture")
    print("="*70)
    print(f"[CONFIG]  {resolved}")
    print(f"[DIR]     {exp_dir}")
    print(f"[DEVICE]  {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("")
    
    # Create and run pipeline
    try:
        pipeline = FederatedTrainingPipeline(
            config=config,
            experiment_dir=exp_dir
        )
        
        history = pipeline.train()
        
        # Print summary
        print("\n" + "="*70)
        print("📊 TRAINING SUMMARY")
        print("="*70)
        
        if history.losses_distributed:
            final_loss = history.losses_distributed[-1][1]
            print(f"Final distributed loss: {final_loss:.4f}")
        
        if history.metrics_distributed:
            _, acc_series = _metric_series_from_flwr_history(history.metrics_distributed, 'accuracy')
            _, tl_series = _metric_series_from_flwr_history(history.metrics_distributed, 'test_loss')
            if acc_series:
                print(f"Final validation accuracy: {acc_series[-1]:.4f}")
            if tl_series:
                print(f"Final validation loss: {tl_series[-1]:.4f}")
        
        print(f"\n✅ Results saved to: {exp_dir}")
        print("="*70)
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()