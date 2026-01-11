"""
Quick script to check for NaN/Inf in processed Amazon data
"""
import pandas as pd
import numpy as np
from pathlib import Path

def check_client_data(client_id=0):
    data_path = Path(f"data/amazon_2023_processed/client_{client_id}/data.pkl")
    
    if not data_path.exists():
        print(f"❌ Data not found: {data_path}")
        return
    
    print(f"📂 Checking: {data_path}")
    df = pd.read_pickle(data_path)
    
    print(f"\n📊 Dataset info:")
    print(f"   Total rows: {len(df)}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Check each column for NaN/Inf
    print(f"\n🔍 Checking for NaN/Inf:")
    
    for col in df.columns:
        if col in ['text_embedding', 'image_embedding', 'behavior_features']:
            # These are arrays (could be lists or numpy arrays)
            arrays = df[col].values
            
            # Convert to numpy if needed and check
            np_arrays = []
            for arr in arrays:
                if isinstance(arr, list):
                    arr = np.array(arr)
                np_arrays.append(arr)
            
            # Check if any array contains NaN
            has_nan = any(np.isnan(arr).any() for arr in np_arrays)
            has_inf = any(np.isinf(arr).any() for arr in np_arrays)
            
            if has_nan or has_inf:
                print(f"   ❌ {col}: NaN={has_nan}, Inf={has_inf}")
                
                # Find first problematic row
                for idx, arr in enumerate(np_arrays):
                    if np.isnan(arr).any() or np.isinf(arr).any():
                        print(f"      → First issue at row {idx}")
                        print(f"      → Array shape: {arr.shape}")
                        print(f"      → NaN count: {np.isnan(arr).sum()}")
                        print(f"      → Inf count: {np.isinf(arr).sum()}")
                        print(f"      → Min: {np.nanmin(arr) if not np.isnan(arr).all() else 'all NaN'}")
                        print(f"      → Max: {np.nanmax(arr) if not np.isnan(arr).all() else 'all NaN'}")
                        break
            else:
                # Get statistics
                all_values = np.concatenate([arr.flatten() for arr in np_arrays])
                print(f"   ✅ {col}: OK")
                print(f"      → Range: [{all_values.min():.4f}, {all_values.max():.4f}]")
                print(f"      → Mean: {all_values.mean():.4f}, Std: {all_values.std():.4f}")
        else:
            # Regular column
            has_nan = df[col].isna().any()
            has_inf = np.isinf(df[col]).any() if df[col].dtype in [np.float32, np.float64] else False
            
            if has_nan or has_inf:
                print(f"   ❌ {col}: NaN={has_nan}, Inf={has_inf}")
            else:
                print(f"   ✅ {col}: OK")
    
    # Check labels
    print(f"\n🏷️  Label distribution:")
    print(df['label'].value_counts().sort_index())
    
    return df

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 CHECKING AMAZON DATA FOR NaN/Inf")
    print("=" * 60)
    
    # Check first 3 clients
    for client_id in range(3):
        print(f"\n{'=' * 60}")
        df = check_client_data(client_id)
        if df is None:
            break
        print()
    
    print("=" * 60)
    print("✅ Check complete!")

