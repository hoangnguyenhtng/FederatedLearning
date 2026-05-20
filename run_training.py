"""
run_training.py — Full Federated Learning Training
Chạy file này để train toàn bộ 100 rounds với 220K samples, 40 clients.
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import warnings
warnings.filterwarnings("ignore")
os.environ["RAY_DEDUP_LOGS"] = "1"
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"

import yaml
import torch

print("=" * 70)
print("  FEDERATED MULTI-MODAL RECOMMENDATION — FULL TRAINING")
print("  FedPer | Amazon 2023 | 220K samples | 40 clients | 100 rounds")
print("=" * 70)
print(f"  Started at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Device     : {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
print()

cfg_path = project_root / "configs" / "config_multi_category.yaml"
if not cfg_path.exists():
    print(f"❌ Config not found: {cfg_path}")
    sys.exit(1)

with open(cfg_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

from src.training.training_utils import normalize_training_config
normalize_training_config(config)

print("  [Training Settings]")
print(f"   Rounds      : {config['federated']['num_rounds']}")
print(f"   Clients     : {config['federated']['num_clients']}")
print(f"   Client/round: {int(config['federated']['fraction_fit'] * config['federated']['num_clients'])}")
print(f"   Batch size  : {config['training']['batch_size']}")
print(f"   Local epochs: {config['training']['local_epochs']}")
print(f"   Learning rate: {config['training']['learning_rate']}")
print(f"   Data path   : {config['paths']['data_processed']}")
print()

exp_name  = config.get("experiment", {}).get("name", "fedper_multi_category")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_dir   = project_root / "experiments" / f"{exp_name}_{timestamp}"
exp_dir.mkdir(parents=True, exist_ok=True)

print(f"  [Output] {exp_dir}")
print()

from src.training.federated_training_pipeline import FederatedTrainingPipeline

t_start = time.time()
try:
    pipeline = FederatedTrainingPipeline(config=config, experiment_dir=exp_dir)
    print(f"\n  ✅ Initialized in {time.time()-t_start:.1f}s — starting training...\n")

    history = pipeline.train()
    t_total = time.time() - t_start

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total time : {t_total/3600:.2f} hours ({t_total/60:.0f} min)")
    print(f"  Results    : {exp_dir}")

    if history.losses_distributed:
        first_loss = history.losses_distributed[0][1]
        final_loss = history.losses_distributed[-1][1]
        print(f"  Loss       : {first_loss:.4f} → {final_loss:.4f}")

    from src.training.federated_training_pipeline import _metric_series_from_flwr_history
    if history.metrics_distributed:
        _, accs = _metric_series_from_flwr_history(history.metrics_distributed, "accuracy")
        if accs:
            print(f"  Accuracy   : {accs[0]:.4f} → {accs[-1]:.4f}")

    print()
    print("  Next steps:")
    print("  1. python generate_report.py            ← tạo biểu đồ & báo cáo")
    print("  2. python src/api/server.py             ← khởi động API demo")
    print("=" * 70)

except KeyboardInterrupt:
    print(f"\n  ⚠️  Training interrupted by user after {(time.time()-t_start)/60:.1f} min")
    print(f"  Checkpoint saved in: {exp_dir / 'models'}")
except Exception as e:
    print(f"\n  ❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
