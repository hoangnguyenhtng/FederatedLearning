# 🎯 Hướng Dẫn Sử Dụng Amazon Reviews 2023 Dataset

## 📊 TẠI SAO DÙNG AMAZON REVIEWS 2023?

### ✅ Advantages vs Synthetic Data

| Feature | Synthetic (Current) | Amazon 2023 | Improvement |
|---------|---------------------|-------------|-------------|
| **Text embeddings** | Random noise | Real reviews | ♾️ Better |
| **Image features** | 4 dummy values | Real product images | 500x Better |
| **Ratings** | Synthetic | Real user ratings | 100% Real |
| **Data size** | 50K | 701K-23.9M | 14-478x Larger |
| **Expected accuracy** | 30% | 60-75% | 2-2.5x Better |

### 🎯 Perfect Match với Model

```
Your Model Architecture:
✅ Text encoder: 384-dim (SentenceTransformer) 
✅ Image encoder: 2048-dim (ResNet-50)
✅ Behavior features: 32-dim
✅ Output: 5-class rating prediction

Amazon Dataset:
✅ Text: Reviews, descriptions
✅ Images: Product photos
✅ Metadata: Price, ratings, brand
✅ Ratings: 1-5 stars
```

**Perfect fit! 🎉**

---

## 📥 BƯỚC 1: Download Dataset

### Recommended: Start với **All_Beauty** (Small, for testing)

```bash
# Create directory
mkdir -p data/raw/amazon_2023
cd data/raw/amazon_2023

# Download reviews (701K reviews, ~200MB)
wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz

# Download metadata (112K items, ~100MB)
wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_All_Beauty.jsonl.gz

# Extract
gunzip All_Beauty.jsonl.gz
gunzip meta_All_Beauty.jsonl.gz
```

**Estimated time**: 5-10 minutes (depending on internet speed)

---

### Optional: Use **Beauty_and_Personal_Care** (Full dataset, for production)

```bash
# Download reviews (23.9M reviews, ~5GB)
wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Beauty_and_Personal_Care.jsonl.gz

# Download metadata (1M items, ~1GB)
wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Beauty_and_Personal_Care.jsonl.gz
```

**Estimated time**: 30-60 minutes

---

## 🔧 BƯỚC 2: Install Dependencies

```bash
# Activate environment
.\fed_rec_env\Scripts\activate  # Windows

# Install additional packages
pip install sentence-transformers pillow requests
```

---

## ⚙️ BƯỚC 3: Process Data

### Quick Test (10K samples)

```bash
# Run processor (will take ~30-60 minutes for 10K samples)
python src/data_generation/process_amazon_data.py
```

**What it does**:
1. ✅ Loads Amazon reviews & metadata
2. ✅ Encodes text with SentenceTransformer (384-dim)
3. ✅ Downloads images & extracts ResNet-50 features (2048-dim)
4. ✅ Creates behavior features (32-dim)
5. ✅ Splits into 10 federated clients (Non-IID)

**Output**: `data/amazon_2023_processed/client_*/data.pkl`

**Estimated time**:
- 10K samples: ~30-60 minutes
- 100K samples: ~5-8 hours
- Full dataset (701K): ~24-36 hours

---

### Full Dataset

Edit `process_amazon_data.py`:

```python
# Line 371: Remove sample_size limit
SAMPLE_SIZE = None  # Process all data (was: 10000)
```

Then run:

```bash
python src/data_generation/process_amazon_data.py
```

**Note**: For full dataset, recommend running overnight!

---

## 🚀 BƯỚC 4: Update DataLoader

Create new DataLoader for Amazon data:

```python
# src/data_generation/amazon_dataloader.py

import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class AmazonDataset(Dataset):
    """Dataset for Amazon Reviews 2023"""
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: Path to processed pickle file
        """
        self.data = pd.read_pickle(data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        return {
            'user_id': torch.tensor(hash(row['user_id']) % 100000, dtype=torch.long),
            'item_id': torch.tensor(hash(row['item_id']) % 100000, dtype=torch.long),
            'text_embedding': torch.tensor(row['text_embedding'], dtype=torch.float32),
            'image_embedding': torch.tensor(row['image_embedding'], dtype=torch.float32),
            'behavior_features': torch.tensor(row['behavior_features'], dtype=torch.float32),
            'label': torch.tensor(row['label'], dtype=torch.long),
            'rating': torch.tensor(row['rating'], dtype=torch.long)
        }


def get_amazon_dataloaders(
    client_id: int,
    data_dir: str = "data/amazon_2023_processed",
    batch_size: int = 32,
    test_split: float = 0.2
):
    """
    Get train/test DataLoaders for a client
    
    Args:
        client_id: Client ID (0-9)
        data_dir: Directory with processed client data
        batch_size: Batch size
        test_split: Test split ratio
    
    Returns:
        train_loader, test_loader
    """
    # Load client data
    client_path = Path(data_dir) / f"client_{client_id}" / "data.pkl"
    dataset = AmazonDataset(client_path)
    
    # Split train/test
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False  # Don't drop last (LayerNorm handles any batch size)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    return train_loader, test_loader
```

---

## 🔄 BƯỚC 5: Update Training Pipeline

Update `federated_training_pipeline.py` to use Amazon data:

```python
# In _load_data() method:

# OLD:
# from src.data_generation.federated_dataloader import get_federated_dataloaders

# NEW:
from src.data_generation.amazon_dataloader import get_amazon_dataloaders

def _load_data(self):
    """Load Amazon federated dataloaders"""
    logger.info("📂 Loading Amazon federated data...")
    
    data_dir = "data/amazon_2023_processed"
    
    if not Path(data_dir).exists():
        raise FileNotFoundError(
            f"Amazon data not found: {data_dir}\n"
            f"Please run: python src/data_generation/process_amazon_data.py"
        )
    
    dataloaders = {}
    for client_id in range(self.config['federated']['num_clients']):
        try:
            train_loader, test_loader = get_amazon_dataloaders(
                client_id=client_id,
                data_dir=data_dir,
                batch_size=self.config['training']['batch_size']
            )
            dataloaders[client_id] = (train_loader, test_loader)
            logger.info(f"✅ Client {client_id}: {len(train_loader.dataset)} train, "
                       f"{len(test_loader.dataset)} test")
        except Exception as e:
            logger.warning(f"⚠️  Client {client_id} failed: {e}")
    
    return dataloaders
```

---

## 🎯 BƯỚC 6: Update Client Training

Update `src/federated/client.py` to use real embeddings:

```python
# In fit() method, around line 145:

# OLD (DELETE THIS):
# text_emb = torch.randn(batch_size, 384, device=self.device)  # ❌ REMOVE!

# NEW (USE REAL EMBEDDINGS):
if isinstance(batch_data, dict):
    # Use REAL embeddings from Amazon data
    text_emb = batch_data['text_embedding'].to(self.device)      # ✅ REAL!
    image_emb = batch_data['image_embedding'].to(self.device)    # ✅ REAL!
    behavior_feat = batch_data['behavior_features'].to(self.device)
    labels = batch_data['label'].to(self.device)
```

---

## 🚀 BƯỚC 7: Train Model!

```bash
# Run training
python src/training/federated_training_pipeline.py
```

**Expected results with Amazon data**:

```
Round 10: Loss ~1.0, Accuracy ~50% (vs 30% with synthetic)
Round 20: Loss ~0.7, Accuracy ~60%
Round 50: Loss ~0.4, Accuracy ~70-75%
```

**Training time** (All_Beauty, 10K samples):
- CPU: ~30-45 minutes (50 rounds)
- GPU: ~15-20 minutes (50 rounds)

---

## 📊 EXPECTED IMPROVEMENTS

### With All_Beauty (701K samples, 10 clients)

| Metric | Synthetic | Amazon | Improvement |
|--------|-----------|--------|-------------|
| Training Loss | 1.555 | 0.4-0.5 | **3-4x better** |
| Accuracy | 30% | 65-70% | **2.2-2.3x better** |
| Convergence | None | 20-30 rounds | **✅ Converges!** |
| Text quality | Random | Real reviews | **♾️ Better** |
| Image quality | 4 values | ResNet features | **500x Better** |

### With Beauty_and_Personal_Care (23.9M samples)

| Metric | Expected Value |
|--------|----------------|
| Accuracy | **75-80%** |
| NDCG@10 | **0.72-0.78** |
| MRR | **0.68-0.74** |
| Hit Rate@10 | **0.80-0.85** |

---

## 🎓 COMPARISON WITH OTHER DATASETS

### Why Amazon over MovieLens/Yelp?

| Dataset | Reviews | Multi-modal | Categories | Best for |
|---------|---------|-------------|------------|----------|
| **MovieLens** | 20M | ❌ Text only | 1 (movies) | Simple RS |
| **Yelp** | 8M | ✅ Text+Images | Mixed | Restaurant RS |
| **Amazon 2023** | **571M** | ✅ **Text+Images+Meta** | **33** | **Multi-modal FL** |

**Amazon wins** because:
1. ✅ Largest multi-modal dataset
2. ✅ Rich metadata (price, brand, features)
3. ✅ Multiple categories for domain adaptation
4. ✅ Real product images (not user photos)
5. ✅ Standard evaluation splits

---

## 🔍 TROUBLESHOOTING

### Issue 1: Out of Memory during Processing

**Solution**: Process in batches

```python
# In process_amazon_data.py, line 371:
SAMPLE_SIZE = 10000  # Start small
```

### Issue 2: Image Download Slow

**Solution**: Use multiple workers or skip images for testing

```python
# In process_amazon_data.py:
# Skip image download for testing
image_embedding = np.random.randn(2048).astype(np.float32)  # Dummy
```

### Issue 3: CUDA Out of Memory

**Solution**: Reduce batch size

```yaml
# In configs/config.yaml:
training:
  batch_size: 16  # Or even 8
```

---

## 📈 VALIDATION CHECKLIST

After processing Amazon data, verify:

- [ ] Text embeddings are NOT all zeros
  ```python
  assert batch['text_embedding'].std() > 0.1, "Text embeddings too uniform"
  ```

- [ ] Image embeddings are NOT all zeros
  ```python
  assert batch['image_embedding'].std() > 0.1, "Image embeddings too uniform"
  ```

- [ ] Labels in range [0, 4]
  ```python
  assert batch['label'].min() >= 0 and batch['label'].max() <= 4
  ```

- [ ] Loss decreases over rounds
  ```python
  assert history['loss'][10] < history['loss'][0]
  ```

---

## 🎯 NEXT STEPS

### After Amazon Data Works

1. **Experiment with categories**:
   - Try Clothing_Shoes_and_Jewelry
   - Try Electronics
   - Compare performance across domains

2. **Advanced features**:
   - Use review helpfulness for weighting
   - Add temporal features (review time trends)
   - Use category hierarchies

3. **Paper experiments**:
   - Compare with baseline (no personalization)
   - Ablation studies (text-only, image-only, behavior-only)
   - Cross-category transfer learning

---

## 📚 REFERENCES

1. **Amazon Reviews 2023**: https://amazon-reviews-2023.github.io/main.html
2. **Paper**: [Bridging Language and Items for Retrieval and Recommendation](https://arxiv.org/abs/2403.03952)
3. **GitHub**: https://github.com/hyp1231/AmazonReviews2023

---

## ✅ SUCCESS CRITERIA

Amazon data integration thành công khi:

1. ✅ Text embeddings từ real reviews (not random)
2. ✅ Image features từ real product photos
3. ✅ Accuracy > 60% sau 30 rounds
4. ✅ Loss < 0.5 sau 50 rounds
5. ✅ Training curves show clear improvement

---

**Generated by**: AI Assistant  
**Date**: January 5, 2026  
**Dataset**: Amazon Reviews 2023 by McAuley Lab

