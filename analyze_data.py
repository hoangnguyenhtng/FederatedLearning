"""
Script to analyze data quality issues
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

print("="*70)
print("DATA QUALITY ANALYSIS")
print("="*70)

# Load raw data
interactions = pd.read_csv('data/raw/interactions.csv')
items = pd.read_csv('data/raw/items.csv')
users = pd.read_csv('data/raw/users.csv')

print("\n=== 1. DATASET SIZES ===")
print(f"Interactions: {len(interactions):,}")
print(f"Users: {len(users):,}")
print(f"Items: {len(items):,}")

print("\n=== 2. RATING DISTRIBUTION (CRITICAL) ===")
rating_dist = interactions['rating'].value_counts().sort_index()
print(rating_dist)
print("\nPercentages:")
for rating, count in rating_dist.items():
    pct = count / len(interactions) * 100
    print(f"  Rating {rating}: {count:,} ({pct:.2f}%)")

print("\n=== 3. CLASS IMBALANCE RATIO ===")
max_count = rating_dist.max()
min_count = rating_dist.min()
print(f"Max class (rating {rating_dist.idxmax()}): {max_count:,}")
print(f"Min class (rating {rating_dist.idxmin()}): {min_count:,}")
print(f"Imbalance ratio: {max_count/min_count:.1f}:1")
if max_count/min_count > 10:
    print("⚠️  WARNING: SEVERE CLASS IMBALANCE (>10:1)")

print("\n=== 4. DATA SPARSITY ===")
num_users = users['user_id'].nunique()
num_items = items['item_id'].nunique()
num_interactions = len(interactions)
possible_interactions = num_users * num_items
sparsity = 1 - (num_interactions / possible_interactions)
print(f"Possible interactions: {possible_interactions:,}")
print(f"Actual interactions: {num_interactions:,}")
print(f"Sparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")
print(f"Density: {1-sparsity:.6f} ({(1-sparsity)*100:.4f}%)")

print("\n=== 5. INTERACTIONS PER USER ===")
user_counts = interactions['user_id'].value_counts()
print(f"Mean: {user_counts.mean():.2f}")
print(f"Median: {user_counts.median():.2f}")
print(f"Min: {user_counts.min()}")
print(f"Max: {user_counts.max()}")
print(f"Std: {user_counts.std():.2f}")

print("\n=== 6. INTERACTIONS PER ITEM ===")
item_counts = interactions['item_id'].value_counts()
print(f"Mean: {item_counts.mean():.2f}")
print(f"Median: {item_counts.median():.2f}")
print(f"Min: {item_counts.min()}")
print(f"Max: {item_counts.max()}")
print(f"Std: {item_counts.std():.2f}")
print(f"\nItems with NO interactions: {len(items) - len(item_counts)}")

print("\n=== 7. CHECK CLIENT DATA ===")
for client_id in range(10):
    client_path = f'data/simulated_clients/client_{client_id}/interactions.csv'
    if Path(client_path).exists():
        client_df = pd.read_csv(client_path)
        ratings = client_df['rating'].value_counts().sort_index()
        print(f"\nClient {client_id}: {len(client_df)} samples")
        print(f"  Ratings: {dict(ratings)}")
    else:
        print(f"Client {client_id}: NOT FOUND")

print("\n=== 8. CHECK EMBEDDINGS IN CLIENT DATA ===")
# Check first client
client_0 = pd.read_csv('data/simulated_clients/client_0/interactions.csv')
print(f"Client 0 columns: {client_0.columns.tolist()}")
print(f"\nSample data:")
print(client_0.head(3))

# Check if embeddings exist
if 'text_embedding' in client_0.columns:
    print("\n✅ Text embeddings found in data")
    # Try to parse first embedding
    try:
        import ast
        first_emb = ast.literal_eval(client_0['text_embedding'].iloc[0])
        print(f"   Text embedding dim: {len(first_emb)}")
    except:
        print("   ⚠️  Could not parse text embedding")
else:
    print("\n❌ No text embeddings in client data")

if 'image_embedding' in client_0.columns:
    print("✅ Image embeddings found in data")
else:
    print("❌ No image embeddings in client data")

if 'behavior_features' in client_0.columns:
    print("✅ Behavior features found in data")
else:
    print("❌ No behavior features in client data")

print("\n=== 9. CHECK LABEL ENCODING ===")
print(f"Unique ratings in raw data: {sorted(interactions['rating'].unique())}")
print(f"Unique ratings in client 0: {sorted(client_0['rating'].unique())}")
print(f"Expected for 5-class: [1, 2, 3, 4, 5] or [0, 1, 2, 3, 4]")

# Check if labels need to be converted
if client_0['rating'].min() == 1:
    print("⚠️  Labels are 1-5, need to convert to 0-4 for PyTorch!")

print("\n=== 10. SUMMARY & ISSUES ===")
issues = []

# Check imbalance
if max_count/min_count > 10:
    issues.append(f"SEVERE class imbalance ({max_count/min_count:.1f}:1)")

# Check sparsity
if sparsity > 0.99:
    issues.append(f"EXTREME sparsity ({sparsity*100:.2f}%)")

# Check data size
if num_interactions < 100000:
    issues.append(f"Small dataset ({num_interactions:,} samples)")

# Check items without data
items_no_data = len(items) - len(item_counts)
if items_no_data > 0:
    issues.append(f"{items_no_data} items with NO interactions (cold start)")

if issues:
    print("❌ ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("✅ No major issues detected")

print("\n" + "="*70)

