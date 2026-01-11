"""
Check Data Distribution for Multi-Category Federated Learning

This script analyzes the distribution of processed Amazon multi-category data:
- Number of samples per client
- Users and items per client
- Rating distribution
- Category distribution
- Non-IID statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def load_client_data(data_dir: Path, client_id: int) -> pd.DataFrame:
    """Load client data from pickle file"""
    client_path = data_dir / f"client_{client_id}" / "data.pkl"
    
    if not client_path.exists():
        return None
    
    try:
        df = pd.read_pickle(client_path)
        return df
    except Exception as e:
        print(f"   ⚠️  Error loading client {client_id}: {e}")
        return None


def analyze_data_distribution(data_dir: str = "data/processed/multi_category"):
    """Analyze distribution of multi-category federated data"""
    
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print(f"   Please run: python src/data_generation/process_amazon_multi_category.py")
        return
    
    print("="*70)
    print("MULTI-CATEGORY DATA DISTRIBUTION ANALYSIS")
    print("="*70)
    print(f"Data directory: {data_dir}\n")
    
    # Find all client directories
    client_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("client_")])
    
    if not client_dirs:
        print(f"❌ No client directories found in {data_dir}")
        return
    
    num_clients = len(client_dirs)
    print(f"📊 Found {num_clients} clients\n")
    
    # Load metadata if available
    metadata_path = data_dir / "processing_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"📋 Processing metadata:")
        print(f"   Categories: {', '.join(metadata.get('categories', []))}")
        print(f"   Total samples: {metadata.get('total_samples', 'N/A'):,}")
        print(f"   Processing date: {metadata.get('processing_date', 'N/A')}")
        print()
    
    # Collect statistics
    all_stats = []
    
    print("="*70)
    print("PER-CLIENT STATISTICS")
    print("="*70)
    
    for client_dir in client_dirs:
        client_id = int(client_dir.name.split('_')[1])
        df = load_client_data(data_dir, client_id)
        
        if df is None or len(df) == 0:
            print(f"\n⚠️  Client {client_id}: No data found")
            continue
        
        # Basic statistics
        num_samples = len(df)
        num_users = df['user_id'].nunique()
        num_items = df['item_id'].nunique()
        
        # Rating distribution
        rating_dist = df['rating'].value_counts().sort_index()
        rating_pct = df['rating'].value_counts(normalize=True).sort_index() * 100
        
        # Category distribution
        if 'category' in df.columns:
            category_dist = df['category'].value_counts().to_dict()
            categories_str = ", ".join([f"{k}: {v}" for k, v in category_dist.items()])
        else:
            category_dist = {}
            categories_str = "N/A"
        
        # Store statistics
        stats = {
            'client_id': client_id,
            'num_samples': num_samples,
            'num_users': num_users,
            'num_items': num_items,
            'avg_samples_per_user': num_samples / num_users if num_users > 0 else 0,
            'rating_dist': rating_dist.to_dict(),
            'category_dist': category_dist
        }
        all_stats.append(stats)
        
        # Print summary
        print(f"\n{'─'*70}")
        print(f"Client {client_id}:")
        print(f"   Samples: {num_samples:,}")
        print(f"   Users: {num_users:,}")
        print(f"   Items: {num_items:,}")
        print(f"   Avg samples/user: {stats['avg_samples_per_user']:.1f}")
        
        if category_dist:
            print(f"   Categories: {categories_str}")
        
        print(f"   Rating distribution:")
        for rating in sorted(rating_dist.index):
            count = rating_dist[rating]
            pct = rating_pct[rating]
            print(f"      Rating {rating}: {count:,} ({pct:.1f}%)")
    
    # Overall statistics
    print(f"\n{'='*70}")
    print("OVERALL STATISTICS")
    print(f"{'='*70}\n")
    
    if not all_stats:
        print("❌ No valid client data found")
        return
    
    total_samples = sum(s['num_samples'] for s in all_stats)
    total_users = set()
    total_items = set()
    all_ratings = Counter()
    all_categories = Counter()
    
    for client_dir in client_dirs:
        client_id = int(client_dir.name.split('_')[1])
        df = load_client_data(data_dir, client_id)
        if df is not None and len(df) > 0:
            total_users.update(df['user_id'].unique())
            total_items.update(df['item_id'].unique())
            all_ratings.update(df['rating'].value_counts().to_dict())
            if 'category' in df.columns:
                all_categories.update(df['category'].value_counts().to_dict())
    
    print(f"Total samples: {total_samples:,}")
    print(f"Total unique users: {len(total_users):,}")
    print(f"Total unique items: {len(total_items):,}")
    print(f"Average samples per client: {total_samples / num_clients:,.0f}")
    print(f"Std dev samples per client: {np.std([s['num_samples'] for s in all_stats]):,.0f}")
    
    print(f"\nOverall rating distribution:")
    for rating in sorted(all_ratings.keys()):
        count = all_ratings[rating]
        pct = count / total_samples * 100
        print(f"   Rating {rating}: {count:,} ({pct:.1f}%)")
    
    if all_categories:
        print(f"\nOverall category distribution:")
        for category in sorted(all_categories.keys()):
            count = all_categories[category]
            pct = count / total_samples * 100
            print(f"   {category}: {count:,} ({pct:.1f}%)")
    
    # Non-IID analysis
    print(f"\n{'='*70}")
    print("NON-IID ANALYSIS")
    print(f"{'='*70}\n")
    
    sample_counts = [s['num_samples'] for s in all_stats]
    cv = np.std(sample_counts) / np.mean(sample_counts) if np.mean(sample_counts) > 0 else 0
    
    print(f"Coefficient of Variation (CV): {cv:.3f}")
    print(f"   CV < 0.1: Very IID (uniform)")
    print(f"   CV 0.1-0.5: Moderate Non-IID")
    print(f"   CV > 0.5: Highly Non-IID")
    
    min_samples = min(sample_counts)
    max_samples = max(sample_counts)
    ratio = max_samples / min_samples if min_samples > 0 else 0
    
    print(f"\nSample size range:")
    print(f"   Min: {min_samples:,}")
    print(f"   Max: {max_samples:,}")
    print(f"   Ratio (max/min): {ratio:.2f}x")
    
    # Category-based analysis
    if all_categories:
        print(f"\n{'='*70}")
        print("CATEGORY-BASED CLIENT DISTRIBUTION")
        print(f"{'='*70}\n")
        
        # Group clients by category
        category_clients = {}
        for stats in all_stats:
            client_id = stats['client_id']
            df = load_client_data(data_dir, client_id)
            if df is not None and 'category' in df.columns:
                main_category = df['category'].mode()[0] if len(df['category'].mode()) > 0 else 'Unknown'
                if main_category not in category_clients:
                    category_clients[main_category] = []
                category_clients[main_category].append(client_id)
        
        for category, client_ids in sorted(category_clients.items()):
            print(f"{category}: {len(client_ids)} clients")
            print(f"   Client IDs: {client_ids}")
    
    # Visualization
    try:
        print(f"\n{'='*70}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*70}\n")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Samples per client
        client_ids = [s['client_id'] for s in all_stats]
        sample_counts = [s['num_samples'] for s in all_stats]
        axes[0, 0].bar(client_ids, sample_counts, color='steelblue')
        axes[0, 0].set_title('Samples per Client')
        axes[0, 0].set_xlabel('Client ID')
        axes[0, 0].set_ylabel('Number of Samples')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Users per client
        user_counts = [s['num_users'] for s in all_stats]
        axes[0, 1].bar(client_ids, user_counts, color='coral')
        axes[0, 1].set_title('Users per Client')
        axes[0, 1].set_xlabel('Client ID')
        axes[0, 1].set_ylabel('Number of Users')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Rating distribution (stacked bar)
        rating_df = pd.DataFrame([s['rating_dist'] for s in all_stats], index=client_ids)
        rating_df = rating_df.fillna(0)
        rating_df.plot(kind='bar', stacked=True, ax=axes[1, 0], colormap='viridis')
        axes[1, 0].set_title('Rating Distribution per Client')
        axes[1, 0].set_xlabel('Client ID')
        axes[1, 0].set_ylabel('Number of Samples')
        axes[1, 0].legend(title='Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Category distribution (if available)
        if all_categories:
            category_df = pd.DataFrame([s['category_dist'] for s in all_stats], index=client_ids)
            category_df = category_df.fillna(0)
            category_df.plot(kind='bar', stacked=True, ax=axes[1, 1], colormap='Set3')
            axes[1, 1].set_title('Category Distribution per Client')
            axes[1, 1].set_xlabel('Client ID')
            axes[1, 1].set_ylabel('Number of Samples')
            axes[1, 1].legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No category data', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Category Distribution (N/A)')
        
        plt.tight_layout()
        
        output_path = data_dir / "distribution_visualization.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to: {output_path}")
        
    except Exception as e:
        print(f"⚠️  Could not generate visualization: {e}")
        print(f"   (This is OK if matplotlib is not installed)")
    
    print(f"\n{'='*70}")
    print("✅ ANALYSIS COMPLETE!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check data distribution for multi-category federated learning")
    parser.add_argument('--data_dir', type=str, default='data/processed/multi_category',
                       help='Path to processed data directory')
    args = parser.parse_args()
    
    analyze_data_distribution(args.data_dir)
