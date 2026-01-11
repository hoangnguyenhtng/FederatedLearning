"""
Federated DataLoader
====================

Creates PyTorch DataLoaders for federated clients with multi-modal data.
Handles text, image, and behavior features.

Author: Federated Multi-Modal Recommendation System
Date: 2024
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import pickle


class MultiModalDataset(Dataset):
    """
    Multi-modal dataset for recommendation
    
    Features:
    - Text descriptions (raw text)
    - Image features (pre-extracted or loaded)
    - Behavior features (user-item interaction features)
    - Ratings (target labels)
    """
    
    def __init__(
        self,
        interactions_df: pd.DataFrame,
        items_df: pd.DataFrame,
        users_df: pd.DataFrame,
        transform=None
    ):
        """
        Initialize multi-modal dataset
        
        Args:
            interactions_df: DataFrame with user_id, item_id, rating
            items_df: DataFrame with item metadata (text, image features)
            users_df: DataFrame with user metadata
            transform: Optional transform to apply
        """
        self.interactions = interactions_df.reset_index(drop=True)
        self.items = items_df
        self.users = users_df
        self.transform = transform
        
        # Create item lookup dictionary for fast access
        self.item_dict = self.items.set_index('item_id').to_dict('index')
        
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        row = self.interactions.iloc[idx]
        
        user_id = row['user_id']
        item_id = row['item_id']
        rating = row['rating']
        
        # Get item features
        item_data = self.item_dict[item_id]
        
        # Text description
        text = item_data.get('description', item_data.get('title', ''))
        
        # Image features (assuming pre-extracted)
        if 'image_features' in item_data:
            if isinstance(item_data['image_features'], str):
                # If stored as string, parse it
                # Use np.fromstring with proper handling (deprecated but works)
                # Or use eval if it's a Python list string
                try:
                    # Try eval first (if it's a Python list string like "[1.0, 2.0, ...]")
                    image_features = np.array(eval(item_data['image_features']), dtype=np.float32)
                except:
                    # Fallback: use np.fromstring with sep (deprecated but works)
                    # Or better: use np.loadtxt for comma-separated values
                    try:
                        # Try np.loadtxt first (better than fromstring)
                        image_features = np.loadtxt(
                            [item_data['image_features'].strip('[]')],
                            delimiter=',',
                            dtype=np.float32
                        )
                    except:
                        # Last fallback: manual parsing
                        try:
                            # Remove brackets and split by comma
                            cleaned = item_data['image_features'].strip('[]').strip()
                            if cleaned:
                                image_features = np.array([float(x.strip()) for x in cleaned.split(',')], dtype=np.float32)
                            else:
                                raise ValueError("Empty image features")
                        except:
                            # Last resort: create dummy features
                            image_features = np.random.randn(512).astype(np.float32)
            elif isinstance(item_data['image_features'], (list, np.ndarray)):
                image_features = np.array(item_data['image_features'], dtype=np.float32)
                # Ensure it's not empty
                if image_features.size == 0:
                    image_features = np.random.randn(512).astype(np.float32)
                # Ensure it's 1D
                if len(image_features.shape) > 1:
                    image_features = image_features.flatten()
            else:
                # Unknown type, use dummy
                image_features = np.random.randn(512).astype(np.float32)
        else:
            # Dummy image features if not available
            image_features = np.random.randn(512).astype(np.float32)
        
        # Final validation: ensure image_features is not empty and has correct shape
        if image_features.size == 0 or len(image_features.shape) == 0:
            image_features = np.random.randn(512).astype(np.float32)
        
        # Ensure it's 1D array with 512 features
        if len(image_features.shape) > 1:
            image_features = image_features.flatten()
        if image_features.shape[0] != 512:
            # Resize or pad to 512
            if image_features.shape[0] < 512:
                # Pad with zeros
                padding = np.zeros(512 - image_features.shape[0], dtype=np.float32)
                image_features = np.concatenate([image_features, padding])
            else:
                # Truncate
                image_features = image_features[:512]
        
        # Behavior features
        # Handle timestamp - convert to numeric if it's a string
        timestamp = row.get('timestamp', 0)
        if isinstance(timestamp, str):
            try:
                # Try to parse datetime string
                from datetime import datetime
                dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                timestamp = float(dt.timestamp())  # Convert to Unix timestamp
            except (ValueError, TypeError):
                # If parsing fails, use a default numeric value
                timestamp = 0.0
        elif not isinstance(timestamp, (int, float)):
            timestamp = 0.0
        
        # Ensure all values are numeric
        popularity = float(item_data.get('popularity', 0.0))
        avg_rating = float(item_data.get('avg_rating', 3.0))
        num_ratings = float(item_data.get('num_ratings', 0))
        user_feature = float(user_id % 100)
        
        # Create base features (5 features)
        base_features = np.array([
            popularity,
            avg_rating,
            num_ratings,
            float(timestamp),
            user_feature,
        ], dtype=np.float32)
        
        # Expand to 32 features as required by model
        # Strategy: repeat and add derived features
        behavior_features = np.zeros(32, dtype=np.float32)
        
        # Fill first 5 with base features
        behavior_features[:5] = base_features
        
        # Add derived features (interactions, ratios, etc.)
        if num_ratings > 0:
            behavior_features[5] = popularity / (num_ratings + 1)  # Popularity per rating
            behavior_features[6] = avg_rating / 5.0  # Normalized rating
        else:
            behavior_features[5] = 0.0
            behavior_features[6] = 0.0
        
        # Add item metadata features (if available)
        behavior_features[7] = float(item_data.get('price', 0.0)) / 1000.0 if 'price' in item_data else 0.0
        
        # Handle brand: convert string to numeric using hash
        brand_value = item_data.get('brand', '')
        if isinstance(brand_value, str):
            # If brand is a string (e.g., 'Brand_27'), convert to numeric using hash
            brand_numeric = hash(brand_value) % 100
        elif isinstance(brand_value, (int, float)):
            # If brand is already numeric, use it directly
            brand_numeric = float(brand_value) % 100
        else:
            # Default fallback
            brand_numeric = 0.0
        behavior_features[8] = brand_numeric / 100.0
        
        # Add time-based features
        behavior_features[9] = float(timestamp) % 86400 / 86400.0  # Time of day (0-1)
        behavior_features[10] = float(timestamp) % 604800 / 604800.0  # Day of week (0-1)
        
        # Add user-item interaction features
        behavior_features[11] = float(user_id) / 10000.0  # Normalized user ID
        behavior_features[12] = float(item_id) / 10000.0  # Normalized item ID
        behavior_features[13] = float((user_id + item_id) % 100) / 100.0  # Combined feature
        
        # Fill remaining with statistical features and padding
        # Use combinations and transformations of base features (deterministic)
        for i in range(14, 32):
            if i < 20:
                # Statistical features (sin transformations)
                behavior_features[i] = np.sin(base_features[i % 5]) * 0.5 + 0.5
            elif i < 26:
                # More derived features (cos transformations)
                behavior_features[i] = np.cos(base_features[i % 5]) * 0.5 + 0.5
            else:
                # Deterministic padding based on item_id and user_id (not random)
                # This ensures same item-user pair always gets same features
                seed_value = (item_id * 1000 + user_id + i) % 1000
                behavior_features[i] = (seed_value / 1000.0) * 0.1  # Scale to 0-0.1
        
        # Normalize behavior features (avoid division by zero)
        max_val = behavior_features.max()
        min_val = behavior_features.min()
        if max_val > min_val:
            behavior_features = (behavior_features - min_val) / (max_val - min_val + 1e-8)
        # If all zeros, keep as is
        
        # Validate and convert rating to label (0-4 for 5 classes)
        # Rating is 1-5, but model needs 0-4 for classification
        rating_value = int(rating)
        if rating_value < 1:
            rating_value = 1
        elif rating_value > 5:
            rating_value = 5
        label = rating_value - 1  # Convert 1-5 → 0-4
        
        sample = {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'item_id': torch.tensor(item_id, dtype=torch.long),
            'text': text,
            'image_features': torch.tensor(image_features, dtype=torch.float32),
            'behavior_features': torch.tensor(behavior_features, dtype=torch.float32),
            'rating': torch.tensor(rating, dtype=torch.long),  # Keep rating for metadata (1-5)
            # Use rating-1 as label for rating prediction task (0-4 for 5 classes)
            'label': torch.tensor(label, dtype=torch.long)  # Label for model training (0-4)
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class FederatedDataLoader:
    """
    DataLoader for federated client
    
    Loads client-specific data and creates train/test splits
    """
    
    def __init__(
        self,
        client_id: int,
        data_dir: str = './data/simulated_clients',
        batch_size: int = 32,
        test_split: float = 0.2,
        seed: int = 42,
        num_workers: int = 0
    ):
        """
        Initialize federated data loader
        
        Args:
            client_id: Client identifier
            data_dir: Directory containing client data
            batch_size: Batch size for DataLoader
            test_split: Fraction of data for testing
            seed: Random seed
            num_workers: Number of worker threads
        """
        self.client_id = client_id
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.test_split = test_split
        self.seed = seed
        self.num_workers = num_workers
        
        # Load client data
        self.interactions_df, self.items_df, self.users_df = self._load_client_data()
        
    def _load_client_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load data for this client"""
        client_dir = self.data_dir / f'client_{self.client_id}'
        
        if not client_dir.exists():
            raise FileNotFoundError(f"Client directory not found: {client_dir}")
        
        # Load interactions
        interactions_path = client_dir / 'interactions.csv'
        if not interactions_path.exists():
            raise FileNotFoundError(f"Interactions file not found: {interactions_path}")
        
        interactions = pd.read_csv(interactions_path)
        
        # Load global items (shared across all clients)
        items_path = self.data_dir / 'items_global.csv'
        if not items_path.exists():
            # Try parent directory
            items_path = self.data_dir.parent / 'raw' / 'items.csv'
        
        if not items_path.exists():
            raise FileNotFoundError(f"Items file not found: {items_path}")
        
        items = pd.read_csv(items_path)
        
        # Load users
        users_path = client_dir / 'users.csv'
        if not users_path.exists():
            # Try loading from user_ids.txt
            user_ids_path = client_dir / 'user_ids.txt'
            if user_ids_path.exists():
                user_ids = pd.read_csv(user_ids_path, header=None, names=['user_id'])
                # Load full users data
                full_users_path = self.data_dir.parent / 'raw' / 'users.csv'
                if full_users_path.exists():
                    full_users = pd.read_csv(full_users_path)
                    users = full_users[full_users['user_id'].isin(user_ids['user_id'])]
                else:
                    # Create dummy users
                    users = user_ids
            else:
                # Create dummy users from interactions
                unique_users = interactions['user_id'].unique()
                users = pd.DataFrame({'user_id': unique_users})
        else:
            users = pd.read_csv(users_path)
        
        print(f"✓ Client {self.client_id} data loaded:")
        print(f"  - Interactions: {len(interactions)}")
        print(f"  - Items: {len(items)}")
        print(f"  - Users: {len(users)}")
        
        return interactions, items, users
    
    def create_dataloaders(
        self,
        text_encoder=None
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and test DataLoaders
        
        Args:
            text_encoder: Optional text encoder (not used during data loading,
                         but can be passed for consistency)
        
        Returns:
            train_loader, test_loader
        """
        # Split data
        np.random.seed(self.seed)
        
        # Shuffle interactions
        interactions_shuffled = self.interactions_df.sample(
            frac=1,
            random_state=self.seed
        ).reset_index(drop=True)
        
        # Split into train/test
        n_test = int(len(interactions_shuffled) * self.test_split)
        
        test_interactions = interactions_shuffled[:n_test]
        train_interactions = interactions_shuffled[n_test:]
        
        print(f"✓ Data split for Client {self.client_id}:")
        print(f"  - Train: {len(train_interactions)} samples")
        print(f"  - Test: {len(test_interactions)} samples")
        
        # Create datasets
        train_dataset = MultiModalDataset(
            train_interactions,
            self.items_df,
            self.users_df
        )
        
        test_dataset = MultiModalDataset(
            test_interactions,
            self.items_df,
            self.users_df
        )
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True  # Drop last incomplete batch to avoid BatchNorm errors
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True  # Drop last incomplete batch to avoid BatchNorm errors
        )
        
        return train_loader, test_loader
    
    def get_data_statistics(self) -> Dict:
        """Get statistics about the data"""
        stats = {
            'num_interactions': len(self.interactions_df),
            'num_users': self.users_df['user_id'].nunique(),
            'num_items': len(self.items_df),
            'rating_distribution': self.interactions_df['rating'].value_counts().to_dict(),
            'sparsity': 1 - len(self.interactions_df) / (
                self.users_df['user_id'].nunique() * len(self.items_df)
            )
        }
        return stats


def get_federated_dataloaders(
    num_clients: int,
    data_dir: str = './data/simulated_clients',
    batch_size: int = 32,
    test_split: float = 0.2,
    seed: int = 42
) -> List[Tuple[DataLoader, DataLoader]]:
    """
    Get DataLoaders for all federated clients
    
    This is a convenience function to create loaders for multiple clients at once.
    
    Args:
        num_clients: Number of clients
        data_dir: Directory containing client data
        batch_size: Batch size for all clients
        test_split: Test split fraction
        seed: Random seed
    
    Returns:
        List of (train_loader, test_loader) tuples, one per client
        
    Example:
        >>> loaders = get_federated_dataloaders(num_clients=10)
        >>> train_loader_0, test_loader_0 = loaders[0]
    """
    print("=" * 70)
    print(f"Creating DataLoaders for {num_clients} clients")
    print("=" * 70)
    
    all_loaders = []
    
    for client_id in range(num_clients):
        print(f"\nClient {client_id}:")
        
        try:
            fed_loader = FederatedDataLoader(
                client_id=client_id,
                data_dir=data_dir,
                batch_size=batch_size,
                test_split=test_split,
                seed=seed
            )
            
            train_loader, test_loader = fed_loader.create_dataloaders()
            all_loaders.append((train_loader, test_loader))
            
        except FileNotFoundError as e:
            print(f"⚠️  Warning: {e}")
            print(f"   Skipping client {client_id}")
            continue
    
    print("\n" + "=" * 70)
    print(f"✓ Created DataLoaders for {len(all_loaders)} clients")
    print("=" * 70)
    
    return all_loaders


def test_dataloader():
    """Test the dataloader"""
    print("=" * 70)
    print("Testing Federated DataLoader")
    print("=" * 70)
    
    # Test single client
    print("\n1. Testing single client DataLoader:")
    try:
        loader = FederatedDataLoader(
            client_id=0,
            data_dir='./data/simulated_clients',
            batch_size=16,
            test_split=0.2
        )
        
        train_loader, test_loader = loader.create_dataloaders()
        
        # Get one batch
        batch = next(iter(train_loader))
        
        print(f"\n✓ Batch keys: {batch.keys()}")
        print(f"✓ Batch shapes:")
        for key, value in batch.items():
            if torch.is_tensor(value):
                print(f"  - {key}: {value.shape}")
            elif isinstance(value, list):
                print(f"  - {key}: list of {len(value)} items")
        
        # Get statistics
        stats = loader.get_data_statistics()
        print(f"\n✓ Data statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  - {key}: {value}")
            else:
                print(f"  - {key}: {value:.4f}" if isinstance(value, float) else f"  - {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test multiple clients
    print("\n" + "=" * 70)
    print("2. Testing multiple clients:")
    try:
        loaders = get_federated_dataloaders(
            num_clients=3,
            data_dir='./data/simulated_clients',
            batch_size=16
        )
        
        print(f"\n✓ Created loaders for {len(loaders)} clients")
        
    except Exception as e:
        print(f"⚠️  Warning: {e}")
    
    print("\n" + "=" * 70)
    print("✓ DataLoader test complete!")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    # Test the dataloader
    test_dataloader()