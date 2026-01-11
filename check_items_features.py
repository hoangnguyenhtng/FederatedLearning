import pandas as pd
import numpy as np

print("=== CHECK ITEMS FEATURES ===")
items = pd.read_csv('data/simulated_clients/items_global.csv')
print(f"Columns: {items.columns.tolist()}")
print(f"\nTotal items: {len(items)}")

print("\n=== SAMPLE ITEM (first) ===")
print(items.iloc[0])

print("\n=== CHECK FEATURE AVAILABILITY ===")
print(f"Has 'text_keywords': {'text_keywords' in items.columns}")
print(f"Has 'description': {'description' in items.columns}")
print(f"Has 'title': {'title' in items.columns}")
print(f"Has 'name': {'name' in items.columns}")
print(f"Has 'image_features': {'image_features' in items.columns}")

if 'image_features' in items.columns:
    print(f"\n=== IMAGE FEATURES ANALYSIS ===")
    sample_feat = items['image_features'].iloc[0]
    print(f"Type: {type(sample_feat)}")
    print(f"Sample (first 200 chars): {str(sample_feat)[:200]}")
    
    # Try to parse
    try:
        if isinstance(sample_feat, str):
            import ast
            parsed = ast.literal_eval(sample_feat)
            print(f"✅ Successfully parsed as list")
            print(f"   Length: {len(parsed)}")
            print(f"   First 5 values: {parsed[:5]}")
        else:
            print(f"Already numeric: {sample_feat[:5]}")
    except Exception as e:
        print(f"❌ Failed to parse: {e}")

if 'text_keywords' in items.columns:
    print(f"\n=== TEXT FEATURES ===")
    print(f"Sample text: {items['text_keywords'].iloc[0]}")

