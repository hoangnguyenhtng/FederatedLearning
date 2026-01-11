"""
Script kiểm tra imports để debug lỗi

Chạy script này để xác định module nào bị lỗi
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("TESTING IMPORTS - PHASE 1 DEBUG")
print("=" * 70)

# Test 1: Privacy module
print("\n1. Testing privacy module...")
try:
    from src.federated.privacy import (
        apply_differential_privacy,
        compute_privacy_budget,
        get_dp_optimizer
    )
    print("   ✓ Privacy module OK")
except Exception as e:
    print(f"   ✗ Privacy module FAILED: {e}")
    sys.exit(1)

# Test 2: Aggregator module
print("\n2. Testing aggregator module...")
try:
    from src.federated.aggregator import (
        FedAvgAggregator,
        get_aggregation_strategy
    )
    print("   ✓ Aggregator module OK")
except Exception as e:
    print(f"   ✗ Aggregator module FAILED: {e}")
    sys.exit(1)

# Test 3: Server module
print("\n3. Testing server module...")
try:
    from src.federated.server import (
        FedPerStrategy,
        create_fedper_strategy,
        get_initial_parameters
    )
    print("   ✓ Server module OK")
except Exception as e:
    print(f"   ✗ Server module FAILED: {e}")
    sys.exit(1)

# Test 4: Client module
print("\n4. Testing client module...")
try:
    from src.federated.client import FederatedClient, create_client_fn
    print("   ✓ Client module OK")
except Exception as e:
    print(f"   ✗ Client module FAILED: {e}")
    sys.exit(1)

# Test 5: Federated package
print("\n5. Testing federated package import...")
try:
    from src.federated import (
        FederatedClient,
        FedPerStrategy,
        apply_differential_privacy
    )
    print("   ✓ Federated package OK")
except Exception as e:
    print(f"   ✗ Federated package FAILED: {e}")
    sys.exit(1)

# Test 6: Models
print("\n6. Testing models...")
try:
    from src.models.multimodal_encoder import MultiModalEncoder
    from src.models.recommendation_model import FedPerRecommender
    print("   ✓ Models OK")
except Exception as e:
    print(f"   ✗ Models FAILED: {e}")
    sys.exit(1)

# Test 7: Training utils
print("\n7. Testing training utilities...")
try:
    from src.training.training_utils import (
        calculate_metrics,
        MetricsCalculator,
        save_checkpoint,
        load_checkpoint
    )
    print("   ✓ Training utilities OK")
except Exception as e:
    print(f"   ✗ Training utilities FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7b: Local trainer (might have circular import)
print("\n7b. Testing local trainer...")
try:
    from src.training.local_trainer import LocalTrainer
    print("   ✓ Local trainer OK")
except Exception as e:
    print(f"   ⚠ Local trainer warning: {e}")
    print("   (This might be OK if there's a circular import)")

# Test 8: Complete import for training pipeline
print("\n8. Testing complete training pipeline imports...")
try:
    from src.training.federated_training_pipeline import FederatedTrainingPipeline
    print("   ✓ Training pipeline OK")
except Exception as e:
    print(f"   ✗ Training pipeline FAILED: {e}")
    print(f"\n   Detailed error:")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ ALL IMPORTS SUCCESSFUL!")
print("=" * 70)
print("\nYou can now proceed with training.")
print("Run: python src/training/federated_training_pipeline.py")