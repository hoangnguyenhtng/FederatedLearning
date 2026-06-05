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
    config_float,
    config_int,
    normalize_training_config,
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
        normalize_training_config(self.config)
        self.experiment_dir = experiment_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create subdirectories
        self.models_dir = experiment_dir / "models"
        self.logs_dir = experiment_dir / "logs"
        self.metrics_dir = experiment_dir / "metrics"
        
        for dir_path in [self.models_dir, self.logs_dir, self.metrics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.client_states_dir = experiment_dir / "client_states"
        self.client_states_dir.mkdir(parents=True, exist_ok=True)
        self._loader_cache: Dict[int, Tuple] = {}
        
        # ── LAZY LOADING: validate data & store metadata only ──────────
        # Do NOT pre-load all clients into RAM (would be ~4.7 GB) and
        # then have Ray serialize it through the object store (another copy).
        # Instead, each Ray actor loads its own client slice on demand.
        self._setup_data_lazy()
        self.global_model = self._create_model()
        
        logger.info(f"🖥️  Using device: {self.device}")
        logger.info(f"📊 Number of clients: {self.num_clients} | data: {self.data_dir}") 
    
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
        # Support both 'personal_hidden_dims' (list) and 'personal_hidden_dim' (scalar)
        personal_dims = model_config.get('personal_hidden_dims')
        if personal_dims is None:
            phd = model_config.get('personal_hidden_dim', 128)
            personal_dims = [int(phd), int(phd) // 2]
        # Rating prediction: 5 classes (ratings 1-5, mapped to 0-4)
        num_classes = model_config.get('num_classes', 3)  # 3-class sentiment
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
    
    def _setup_data_lazy(self):
        """Validate data exists and store metadata only — NO pre-loading.
        
        Ray actors will load each client's pickle on-demand inside _client_fn,
        so we never serialize 4.7 GB of DataLoaders through the object store.
        """
        logger.info("📂 Validating federated data (lazy mode)...")
        
        self.num_clients  = self.config['federated']['num_clients']
        self.batch_size   = self.config['training']['batch_size']
        self.test_split   = self.config['training'].get('test_split', 0.2)
        self.use_synthetic = False

        amazon_dir = resolve_amazon_federated_data_dir(self.config, cwd=project_root)
        paths_cfg  = self.config.get('paths') or {}
        synthetic_dir = (
            project_root / paths_cfg.get('data_dir', 'data') / 'simulated_clients'
        ).resolve()

        if amazon_dir is not None:
            self.data_dir = amazon_dir
            # Validate all expected client pickles exist
            missing = [
                i for i in range(self.num_clients)
                if not (amazon_dir / f'client_{i}' / 'data.pkl').exists()
            ]
            if missing:
                logger.warning(f"⚠️  Missing client pickles: {missing}")
            found = self.num_clients - len(missing)
            logger.info(f"🎉 Amazon data validated: {found}/{self.num_clients} clients found at {amazon_dir}")
            # Log one client for sanity
            sample_pkl = amazon_dir / 'client_0' / 'data.pkl'
            if sample_pkl.exists():
                import pandas as pd
                df0 = pd.read_pickle(sample_pkl)
                n_train = int(len(df0) * (1 - self.test_split))
                n_test  = len(df0) - n_train
                logger.info(f"   client_0 preview: {len(df0):,} rows → train={n_train:,} / test={n_test:,}")
                logger.info(f"   Columns: {list(df0.columns)}")
                del df0  # release memory immediately

        elif synthetic_dir.exists():
            self.data_dir     = synthetic_dir
            self.use_synthetic = True
            logger.warning("⚠️  No Amazon data found — falling back to SYNTHETIC data")
        else:
            raise FileNotFoundError(
                "❌ No data found!\n"
                "Run: python src/data_generation/process_amazon_multi_category.py "
                "--config configs/config_multi_category.yaml"
            )
    
    def _client_fn(self, context):
        """Create a Flower client — data is loaded lazily inside this actor.
        
        Each Ray actor calls this function independently; only this one client's
        pickle is loaded into that actor's memory, keeping peak RAM ~500 MB/actor
        instead of 4.7 GB shared across the main process.
        """
        # ── Extract client ID ─────────────────────────────────────────
        cid = None
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
        except (ValueError, TypeError, AttributeError) as e:
            raise ValueError(f"Cannot extract client ID from context={context}: {e}")
        cid = int(cid)

        if cid in self._loader_cache:
            train_loader, test_loader = self._loader_cache[cid]
        else:
            train_loader, test_loader = self._load_client_dataloaders(cid)
            self._loader_cache[cid] = (train_loader, test_loader)

        # ── Clone global model for this client ────────────────────────
        import copy
        client_model = copy.deepcopy(self.global_model)
        client_model.to(self.device)

        # ── Pre-load personal head state into model BEFORE creating FedPerClient ──
        # Critical: Without this, deepcopy gives random personal head every round,
        # causing accuracy to plateau because personal head never accumulates knowledge.
        state_path = self.client_states_dir / f"client_{cid}.pt"
        if state_path.exists():
            try:
                payload = torch.load(state_path, map_location=str(self.device), weights_only=False)
                personal = payload.get("personal") or {}
                current = client_model.state_dict()
                for name, param in personal.items():
                    if name in current:
                        current[name] = param.to(self.device)
                client_model.load_state_dict(current, strict=False)
                logger.debug(f"✅ Pre-loaded personal head for client {cid} (round {payload.get('round', '?')})")
            except Exception as e:
                logger.warning(f"⚠️ Could not pre-load personal state for client {cid}: {e}")

        # ── Build Flower client ───────────────────────────────────────
        tr_cfg = self.config["training"]
        fed_cfg = self.config["federated"]
        state_path = self.client_states_dir / f"client_{cid}.pt"
        numpy_client = FedPerClient(
            client_id=cid,
            model=client_model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=str(self.device),
            local_epochs=config_int(tr_cfg.get("local_epochs"), 3),
            learning_rate=config_float(tr_cfg.get("learning_rate"), 1e-3),
            apply_dp=self.config.get('privacy', {}).get('differential_privacy', False),
            weight_decay=config_float(tr_cfg.get("weight_decay"), 1e-4),
            num_rounds=config_int(fed_cfg.get("num_rounds"), 100),
            state_path=state_path,
            personalize_epochs_eval=config_int(tr_cfg.get("personalize_epochs_eval"), 2),
            loss_type=str(tr_cfg.get("loss", "weighted_ce")),
            gradient_clip=config_float(tr_cfg.get("gradient_clip"), 2.0),
            fedprox_mu=config_float(fed_cfg.get("fedprox_mu"), 0.0),
        )
        try:
            client = numpy_client.to_client()
        except AttributeError:
            client = numpy_client

        logger.debug(f"✅ Lazy-loaded client {cid} | train={len(train_loader.dataset):,}")
        return client

    def _load_client_dataloaders(self, cid: int):
        """Load train/test loaders for one client (cached after first call)."""
        import numpy as np
        import pandas as pd
        import torch
        from torch.utils.data import DataLoader

        if not self.use_synthetic:
            pkl_path = self.data_dir / f'client_{cid}' / 'data.pkl'
            if not pkl_path.exists():
                raise FileNotFoundError(f"Client {cid} pickle not found: {pkl_path}")
            client_df = pd.read_pickle(pkl_path)

            client_df = client_df.sample(frac=1, random_state=42 + cid).reset_index(drop=True)
            n_test = int(len(client_df) * self.test_split)
            test_df = client_df[:n_test]
            train_df = client_df[n_test:]

            from src.data_generation.amazon_dataloader import AmazonDataset
            from torch.utils.data import WeightedRandomSampler
            train_ds = AmazonDataset(train_df)
            
            # ✅ WeightedRandomSampler: oversample minority classes
            # Map 5-star ratings to 3-class sentiment: 1-2★→0(Neg), 3★→1(Neu), 4-5★→2(Pos)
            raw_ratings = train_df['rating'].values.astype(int)
            train_labels = np.where(raw_ratings <= 2, 0, np.where(raw_ratings == 3, 1, 2))
            label_counts = np.bincount(train_labels, minlength=3).astype(float)
            label_counts = np.maximum(label_counts, 1.0)  # Avoid division by zero
            sample_weights = 1.0 / label_counts[train_labels]
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(sample_weights).double(),
                num_samples=len(sample_weights),
                replacement=True,
            )
            
            train_loader = DataLoader(
                train_ds, batch_size=self.batch_size,
                sampler=sampler,  # ✅ Replaces shuffle=True
                num_workers=0, pin_memory=False, drop_last=False,
            )
            test_loader = DataLoader(
                AmazonDataset(test_df), batch_size=self.batch_size,
                shuffle=False, num_workers=0, pin_memory=False, drop_last=False
            )
            return train_loader, test_loader

        from src.data_generation.federated_dataloader import get_federated_dataloaders
        loaders_list = get_federated_dataloaders(
            num_clients=self.num_clients, data_dir=self.data_dir,
            batch_size=self.batch_size, test_split=self.test_split
        )
        if cid >= len(loaders_list) or loaders_list[cid] is None:
            raise ValueError(f"Synthetic client {cid} data not available")
        return loaders_list[cid]


    
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

        # Create client function (lazy loading — each client loads data on demand)
        client_fn = self._client_fn

        
        # Configure simulation
        num_rounds = self.config['federated']['num_rounds']
        use_gpu    = torch.cuda.is_available() and self.device.type == 'cuda'
        # Allow up to 5 clients to share the GPU concurrently (0.2 VRAM each)
        gpu_per_client = 0.2 if use_gpu else 0.0
        
        logger.info("Configuration:")
        logger.info(f"  Rounds       : {num_rounds}")
        logger.info(f"  Clients/round: {int(self.config['federated']['fraction_fit'] * self.config['federated']['num_clients'])}")
        logger.info(f"  Local epochs : {self.config['training']['local_epochs']}")
        logger.info(f"  Batch size   : {self.config['training']['batch_size']}")
        logger.info(f"  Learning rate: {self.config['training']['learning_rate']}")
        logger.info(f"  Device       : {self.device} | GPU/client: {gpu_per_client}")
        logger.info(f"  Data mode    : {'Amazon pickle (lazy)' if not self.use_synthetic else 'Synthetic'}")
        logger.info("")

        # Run simulation
        try:
            # Lazy loading means the client_fn closure is tiny (just paths + config).
            # No DataLoader objects are serialised through Ray object store.
            client_resources = {
                "num_cpus": 1,
                "num_gpus": gpu_per_client,
            }

            os.environ.setdefault("RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE", "1")
            os.environ.setdefault("RAY_DEDUP_LOGS", "1")
            # Tat memory monitor cua Ray de tranh false positive tren Windows
            os.environ["RAY_memory_monitor_refresh_ms"] = "0"
            
            # ----------------------------------------------------------------
            # Tinh toan object_store_memory an toan cho Ray
            # ----------------------------------------------------------------
            import psutil
            total_ram     = psutil.virtual_memory().total
            available_ram = psutil.virtual_memory().available

            # Object store: 10% of available RAM, min 256MB, max 512MB
            obj_store_mb  = max(256, min(512, int(available_ram * 0.10 / (1024**2))))
            obj_store_bytes = obj_store_mb * 1024 * 1024

            # _memory: set explicitly to bypass Ray validation check.
            explicit_memory_bytes = min(
                4 * 1024 * 1024 * 1024,           # 4 GB hard cap
                int(total_ram * 0.60),              # 60% of physical RAM
            )

            logger.info(f"RAM total: {total_ram/1024**3:.1f}GB, Available: {available_ram/1024**3:.1f}GB")
            logger.info(f"Ray object_store: {obj_store_mb}MB, task memory: {explicit_memory_bytes/1024**3:.1f}GB")

            ray_init_args = {
                "ignore_reinit_error": True,
                "include_dashboard": False,
                "object_store_memory": obj_store_bytes,
                "_memory": explicit_memory_bytes,
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
            
            # Save results (strategy carries true train_loss per round)
            self._save_results(history, strategy=strategy)
            
            return history
            
        except Exception as e:
            logger.error(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _save_results(self, history, strategy=None):
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
            'metrics_centralized': history.metrics_centralized if history.metrics_centralized else {},
            'note': 'losses_distributed = Flower eval loss (distributed). train_loss_fit = weighted client train loss from fit().',
        }
        if strategy is not None and getattr(strategy, 'metrics_history', None):
            mh = strategy.metrics_history
            history_dict['train_loss_fit'] = list(mh.get('train_loss', []))
            history_dict['test_loss_eval'] = list(mh.get('test_loss', []))
            history_dict['accuracy_eval'] = list(mh.get('accuracy', []))
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        logger.info(f"✅ Saved training history: {history_path}")
        
        # Plot results
        try:
            plot_path = self.metrics_dir / "training_curves.png"
            self._plot_training_history(history, save_path=plot_path, strategy=strategy)
            logger.info(f"✅ Saved training curves: {plot_path}")
        except Exception as e:
            logger.warning(f"⚠️  Could not plot training curves: {e}")
        
        # Save config
        config_path = self.experiment_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        logger.info(f"✅ Saved configuration: {config_path}")
    
    def _plot_training_history(self, history, save_path: Path, strategy=None):
        """Plot training history: Loss và Accuracy theo round"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # === Subplot 1: Loss (eval + optional fit train loss) ===
            if history.losses_distributed:
                rounds_loss, losses = zip(*history.losses_distributed)
                axes[0].plot(rounds_loss, losses, 'b-o', markersize=4, label='Eval loss (Flower)')
            
            if history.losses_centralized:
                rounds_c, losses_c = zip(*history.losses_centralized)
                axes[0].plot(rounds_c, losses_c, 'r--', label='Centralized eval')
            
            if strategy is not None and getattr(strategy, 'metrics_history', None):
                tl_fit = strategy.metrics_history.get('train_loss') or []
                if tl_fit:
                    r_fit, v_fit = zip(*tl_fit)
                    axes[0].plot(r_fit, v_fit, 'c-s', markersize=3, label='Train loss (fit)')
            
            axes[0].set_xlabel('Round')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Loss per round')
            axes[0].legend(fontsize=8)
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
    normalize_training_config(config)

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