"""
Quick test to verify Amazon dataloader works
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*70)
print("TESTING AMAZON DATALOADER")
print("="*70)

# Check if Amazon data exists
amazon_dir = Path("data/amazon_2023_processed")

if not amazon_dir.exists():
    print(f"\n❌ Amazon data not found: {amazon_dir}")
    print(f"\nPlease run:")
    print(f"  python src\\data_generation\\process_amazon_data.py")
    sys.exit(1)

# Check for client data
client_0 = amazon_dir / "client_0" / "data.pkl"
if not client_0.exists():
    print(f"\n❌ Client 0 data not found: {client_0}")
    print(f"\nPlease run:")
    print(f"  python src\\data_generation\\process_amazon_data.py")
    sys.exit(1)

print(f"\n✅ Amazon data found: {amazon_dir}")

# Test loading
try:
    from src.data_generation.amazon_dataloader import get_amazon_dataloaders
    
    print(f"\n📂 Loading dataloaders...")
    loaders = get_amazon_dataloaders(
        num_clients=10,
        data_dir=str(amazon_dir),
        batch_size=16,
        test_split=0.2
    )
    
    print(f"\n✅ Loaded {len(loaders)} clients")
    
    # Test first client
    if 0 in loaders:
        train_loader, test_loader = loaders[0]
        print(f"\n📊 Client 0 stats:")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        
        # Test one batch
        print(f"\n🔍 Testing batch...")
        batch = next(iter(train_loader))
        
        print(f"\n✅ Batch keys: {batch.keys()}")
        print(f"\n   Shapes:")
        for key, value in batch.items():
            if hasattr(value, 'shape'):
                print(f"   - {key}: {value.shape}")
            else:
                print(f"   - {key}: {type(value)}")
        
        # Verify expected keys
        expected_keys = ['text_embedding', 'image_embedding', 'behavior_features', 'label']
        missing_keys = [k for k in expected_keys if k not in batch]
        
        if missing_keys:
            print(f"\n⚠️  Missing keys: {missing_keys}")
        else:
            print(f"\n✅ All expected keys present!")
        
        # Verify shapes
        print(f"\n🔍 Verifying shapes...")
        checks = []
        
        if 'text_embedding' in batch:
            shape = batch['text_embedding'].shape
            checks.append(('text_embedding', shape, shape[1] == 384, '384-dim'))
        
        if 'image_embedding' in batch:
            shape = batch['image_embedding'].shape
            checks.append(('image_embedding', shape, shape[1] == 2048, '2048-dim'))
        
        if 'behavior_features' in batch:
            shape = batch['behavior_features'].shape
            checks.append(('behavior_features', shape, shape[1] == 32, '32-dim'))
        
        if 'label' in batch:
            shape = batch['label'].shape
            min_val = batch['label'].min().item()
            max_val = batch['label'].max().item()
            checks.append(('label', shape, (min_val >= 0 and max_val <= 4), 'range [0-4]'))
        
        all_pass = True
        for name, shape, passed, expected in checks:
            status = "✅" if passed else "❌"
            print(f"   {status} {name}: {shape} (expected {expected})")
            if not passed:
                all_pass = False
        
        if all_pass:
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"\n✅ Ready to train:")
            print(f"   python src\\training\\federated_training_pipeline.py")
        else:
            print(f"\n⚠️  Some checks failed, but may still work")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    print(f"\n💡 Possible fixes:")
    print(f"   1. Re-run: python src\\data_generation\\process_amazon_data.py")
    print(f"   2. Check data files in: {amazon_dir}")

