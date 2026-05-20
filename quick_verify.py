"""Kiểm tra nhanh pipeline (multi_category, lazy-load)."""

import sys
import os
import time
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import warnings
warnings.filterwarnings("ignore")
os.environ["RAY_DEDUP_LOGS"] = "1"
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"

import yaml
import torch

print("=" * 65)
print("  QUICK PIPELINE VERIFICATION  [lazy-load v2]")
print("  Config: config_multi_category.yaml | 220K samples / 40 clients")
print("=" * 65)

cfg_path = project_root / "configs" / "config_multi_category.yaml"
with open(cfg_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

from src.training.training_utils import normalize_training_config
normalize_training_config(config)

# Patch for quick 3-round run
config["federated"]["num_rounds"]            = 3
config["federated"]["num_clients"]           = 40
config["federated"]["fraction_fit"]          = 0.125   # 5/40 clients
config["federated"]["fraction_evaluate"]     = 0.125
config["federated"]["min_fit_clients"]       = 5
config["federated"]["min_evaluate_clients"]  = 5
config["federated"]["min_available_clients"] = 40
config["training"]["local_epochs"]           = 1       # fast test

print(f"\n[Config]")
print(f"  Data path : {config['paths']['data_processed']}")
print(f"  Clients   : {config['federated']['num_clients']}")
print(f"  Rounds    : {config['federated']['num_rounds']} (quick test, 3 rounds)")
print(f"  GPU       : {'CUDA available ✅' if torch.cuda.is_available() else 'CPU only ⚠️'}")
if torch.cuda.is_available():
    print(f"  GPU name  : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n[Data Check — no pre-loading]")
from src.training.training_utils import resolve_amazon_federated_data_dir
data_dir = resolve_amazon_federated_data_dir(config, cwd=project_root)
if data_dir is None:
    print("  ❌ Cannot find processed data!")
    sys.exit(1)

import os as _os
found = sum(
    1 for i in range(40)
    if (data_dir / f"client_{i}" / "data.pkl").exists()
)
total_gb = sum(
    _os.path.getsize(data_dir / f"client_{i}" / "data.pkl")
    for i in range(40)
    if (data_dir / f"client_{i}" / "data.pkl").exists()
) / 1e9

print(f"  ✅ {found}/40 client pickles found | Total size: {total_gb:.1f} GB")
print(f"  ⚡ Lazy mode: each Ray actor loads its own pickle on demand")
print(f"     Peak RAM during training: ~{total_gb/40*6:.1f} GB (6 actors × ~{total_gb/40:.2f} GB/client)")

print("\n[Starting 3-Round FL Simulation — lazy loading...]")
print("  (Startup should be fast now — no 5-minute pre-load)")
print()

from src.training.federated_training_pipeline import FederatedTrainingPipeline
from datetime import datetime

exp_dir = project_root / "experiments" / f"quick_verify_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
exp_dir.mkdir(parents=True, exist_ok=True)

t0 = time.time()
try:
    pipeline = FederatedTrainingPipeline(config=config, experiment_dir=exp_dir)
    t_init = time.time() - t0
    print(f"\n  ✅ Pipeline initialized in {t_init:.1f}s  (was ~341s before fix)")

    t1 = time.time()
    history = pipeline.train()
    t_train = time.time() - t1
    t_total = time.time() - t0

    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  Init time             : {t_init:.1f}s")
    print(f"  3 rounds training     : {t_train:.1f}s")
    print(f"  Time per round        : {t_train/3:.1f}s")

    est_100 = (t_train / 3) * 100
    est_hrs = est_100 / 3600
    print(f"\n  📊 Full 100-round estimate:")
    print(f"     ≈ {est_100/60:.0f} min  ({est_hrs:.1f} hours)")

    if history.losses_distributed:
        losses = [f"{l:.4f}" for _, l in history.losses_distributed]
        print(f"\n  Loss per round: {losses}")

    if history.metrics_distributed:
        from src.training.federated_training_pipeline import _metric_series_from_flwr_history
        rds, accs = _metric_series_from_flwr_history(history.metrics_distributed, "accuracy")
        if accs:
            print(f"  Accuracy per round: {[f'{a:.4f}' for a in accs]}")

    print(f"\n  ✅ PIPELINE VERIFIED — no MemoryError")
    print(f"  Results saved to: {exp_dir}")

except Exception as e:
    print(f"\n  ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 65)
print("  READY FOR FULL TRAINING:")
print("  python run_training.py")
print("  (or)")
print("  python src/training/federated_training_pipeline.py \\")
print("         --config configs/config_multi_category.yaml")
print("=" * 65)
