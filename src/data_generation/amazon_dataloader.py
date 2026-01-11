"""
DataLoader for Amazon Reviews 2023 processed data
Compatible with FederatedTrainingPipeline
"""

import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Dict
import numpy as np


class AmazonDataset(Dataset):
    """Dataset for Amazon Reviews 2023"""
    
    def __init__(self, data_df: pd.DataFrame):
        """
        Args:
            data_df: DataFrame with processed Amazon data
        """
        self.data = data_df.reset_index(drop=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Convert to tensors
        text_emb = torch.tensor(row['text_embedding'], dtype=torch.float32)
        image_emb = torch.tensor(row['image_embedding'], dtype=torch.float32)
        behavior_feat = torch.tensor(row['behavior_features'], dtype=torch.float32)
        
        # SAFETY: Replace any NaN/Inf with 0.0 (backup protection)
        text_emb = torch.nan_to_num(text_emb, nan=0.0, posinf=0.0, neginf=0.0)
        image_emb = torch.nan_to_num(image_emb, nan=0.0, posinf=0.0, neginf=0.0)
        behavior_feat = torch.nan_to_num(behavior_feat, nan=0.0, posinf=0.0, neginf=0.0)
        
        return {
            'user_id': torch.tensor(hash(str(row['user_id'])) % 100000, dtype=torch.long),
            'item_id': torch.tensor(hash(str(row['item_id'])) % 100000, dtype=torch.long),
            'text_embedding': text_emb,
            'image_embedding': image_emb,
            'behavior_features': behavior_feat,
            'label': torch.tensor(row['label'], dtype=torch.long),
            'rating': torch.tensor(row['rating'], dtype=torch.long)
        }


def get_amazon_dataloaders(
    num_clients: int = 10,
    data_dir: str = "data/amazon_2023_processed",
    batch_size: int = 32,
    test_split: float = 0.2,
    seed: int = 42
) -> Dict[int, Tuple[DataLoader, DataLoader]]:
    """
    Get DataLoaders for all Amazon federated clients
    
    Compatible with get_federated_dataloaders() signature
    
    Args:
        num_clients: Number of clients (0 to num_clients-1)
        data_dir: Directory with processed client data
        batch_size: Batch size
        test_split: Test split ratio
        seed: Random seed
    
    Returns:
        Dictionary {client_id: (train_loader, test_loader)}
    """
    print("="*70)
    print(f"Loading Amazon data for {num_clients} clients")
    print("="*70)
    
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(
            f"❌ Amazon data not found: {data_dir}\n"
            f"Please run: python src\\data_generation\\process_amazon_data.py"
        )
    
    all_loaders = {}
    
    for client_id in range(num_clients):
        client_path = data_dir / f"client_{client_id}" / "data.pkl"
        
        if not client_path.exists():
            print(f"⚠️  Client {client_id}: No data found at {client_path}")
            continue
        
        try:
            # Load client data
            print(f"\nClient {client_id}:")
            client_df = pd.read_pickle(client_path)
            
            # Split train/test
            np.random.seed(seed + client_id)
            n_test = int(len(client_df) * test_split)
            
            # Shuffle
            client_df = client_df.sample(frac=1, random_state=seed + client_id).reset_index(drop=True)
            
            test_df = client_df[:n_test]
            train_df = client_df[n_test:]
            
            print(f"  Train: {len(train_df)} samples")
            print(f"  Test: {len(test_df)} samples")
            
            # Create datasets
            train_dataset = AmazonDataset(train_df)
            test_dataset = AmazonDataset(test_df)
            
            # Create DataLoaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # Windows compatibility
                pin_memory=False,
                drop_last=False  # LayerNorm handles any batch size
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                drop_last=False
            )
            
            all_loaders[client_id] = (train_loader, test_loader)
            
        except Exception as e:
            print(f"❌ Client {client_id} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_loaders:
        raise ValueError(
            f"No client data loaded!\n"
            f"Expected data in: {data_dir}/client_*/data.pkl\n"
            f"Please run: python src\\data_generation\\process_amazon_data.py"
        )
    
    print(f"\n✅ Loaded {len(all_loaders)} clients (expected {num_clients})")
    return all_loaders


# Test
if __name__ == "__main__":
    print("Testing Amazon DataLoader...")
    
    try:
        loaders = get_amazon_dataloaders(
            num_clients=10,
            data_dir="data/amazon_2023_processed",
            batch_size=16
        )
        
        # Test first client
        if 0 in loaders:
            train_loader, test_loader = loaders[0]
            print(f"\n✅ Test successful!")
            print(f"   Train batches: {len(train_loader)}")
            print(f"   Test batches: {len(test_loader)}")
            
            # Test one batch
            batch = next(iter(train_loader))
            print(f"\n   Sample batch shapes:")
            print(f"   - text_embedding: {batch['text_embedding'].shape}")
            print(f"   - image_embedding: {batch['image_embedding'].shape}")
            print(f"   - behavior_features: {batch['behavior_features'].shape}")
            print(f"   - label: {batch['label'].shape}")
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print(f"\n💡 To fix:")
        print(f"   1. Download Amazon data: PowerShell -ExecutionPolicy Bypass -File download_amazon_data.ps1")
        print(f"   2. Process data: python src\\data_generation\\process_amazon_data.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

