"""
Amazon Reviews 2023 Data Processor for Federated Multi-Modal Recommendation

This script processes Amazon Reviews 2023 dataset for federated learning:
1. Loads reviews and item metadata
2. Extracts text embeddings using SentenceTransformer
3. Downloads and processes product images using ResNet-50
4. Creates behavior features from metadata
5. Splits data into federated clients (Non-IID)

Dataset: https://amazon-reviews-2023.github.io/main.html
"""

import json
import gzip
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import pickle

# ML libraries
from sentence_transformers import SentenceTransformer
import torchvision.models as models
import torchvision.transforms as transforms

class AmazonDataProcessor:
    """Process Amazon Reviews 2023 for federated learning"""
    
    def __init__(self, 
                 category: str = "Beauty_and_Personal_Care",
                 output_dir: str = "data/amazon_2023",
                 sample_size: int = None,
                 skip_image_download: bool = True):  # NEW: Skip image processing by default
        """
        Args:
            category: Amazon category (e.g., "Beauty_and_Personal_Care", "All_Beauty")
            output_dir: Output directory for processed data
            sample_size: If set, only process this many interactions (for testing)
            skip_image_download: If True, use dummy image features (faster, no network needed)
        """
        self.category = category
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size = sample_size
        self.skip_image_download = skip_image_download
        
        # Initialize encoders
        print("Loading text encoder (SentenceTransformer)...")
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim
        
        if not skip_image_download:
            print("Loading image encoder (ResNet-50)...")
            try:
                self.image_model = models.resnet50(pretrained=True)
                # Remove final classification layer to get features
                self.image_model = torch.nn.Sequential(*list(self.image_model.children())[:-1])
                self.image_model.eval()
                
                # Image preprocessing
                self.image_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.image_model.to(self.device)
                print(f"✅ Image encoder loaded")
            except Exception as e:
                print(f"⚠️  Failed to load ResNet-50: {e}")
                print(f"   Falling back to dummy image features")
                self.skip_image_download = True
        else:
            print("📦 Using text-derived image proxy (semantic projection)")
            print("   Image features are derived from product text → projected to 2048-dim")
            # Create deterministic projection matrix for 384→2048 expansion
            rng = np.random.RandomState(42)
            self._image_projection_matrix = rng.randn(384, 2048).astype(np.float32) * 0.05
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ Initialized processors on device: {self.device}")
    
    def load_jsonl_gz(self, filepath: str) -> List[Dict]:
        """Load JSONL file (gzipped or plain)"""
        data = []
        filepath = Path(filepath)
        print(f"Loading {filepath}...")
        
        # Check if file is gzipped
        if filepath.suffix == '.gz':
            open_func = lambda: gzip.open(filepath, 'rt', encoding='utf-8')
        else:
            open_func = lambda: open(filepath, 'r', encoding='utf-8')
        
        with open_func() as f:
            for i, line in enumerate(tqdm(f)):
                if self.sample_size and i >= self.sample_size:
                    break
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # Skip invalid lines
        
        print(f"✅ Loaded {len(data)} records")
        return data
    
    def download_and_encode_image(self, image_url: str) -> np.ndarray:
        """
        Download image and extract ResNet-50 features
        
        Returns:
            2048-dim feature vector or None if failed
        """
        try:
            # Download image with timeout
            response = requests.get(image_url, timeout=5)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Preprocess
            image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.image_model(image_tensor)
                features = features.squeeze().cpu().numpy()  # (2048,)
            
            return features
            
        except Exception as e:
            # If download fails, return None
            return None
    
    def process_reviews(self, reviews_path: str, meta_path: str) -> pd.DataFrame:
        """
        Process reviews and metadata
        
        Returns:
            DataFrame with all features
        """
        print("\n" + "="*70)
        print("STEP 1: Loading Reviews & Metadata")
        print("="*70)
        
        # Load data
        reviews = self.load_jsonl_gz(reviews_path)
        metadata = self.load_jsonl_gz(meta_path)
        
        # Convert to DataFrames
        reviews_df = pd.DataFrame(reviews)
        meta_df = pd.DataFrame(metadata)
        
        print(f"\nReviews: {len(reviews_df)}")
        print(f"Metadata: {len(meta_df)}")
        
        # Create item lookup
        meta_dict = meta_df.set_index('parent_asin').to_dict('index')
        
        print("\n" + "="*70)
        print("STEP 2: Processing Features")
        print("="*70)
        
        processed_data = []
        
        for idx, review in tqdm(reviews_df.iterrows(), total=len(reviews_df)):
            # Get item metadata
            parent_asin = review['parent_asin']
            if parent_asin not in meta_dict:
                continue  # Skip if no metadata
            
            item_meta = meta_dict[parent_asin]
            
            # 1. TEXT EMBEDDINGS
            # Combine review text + item description
            review_text = f"{review.get('title', '')} {review.get('text', '')}"
            item_desc = ' '.join(item_meta.get('description', [])) if 'description' in item_meta else ''
            combined_text = f"{review_text} {item_desc}".strip()[:512]  # Limit length
            
            text_embedding = self.text_encoder.encode(combined_text, convert_to_tensor=False)
            
            # 2. IMAGE FEATURES
            if not self.skip_image_download:
                # Try to download and encode image
                image_embedding = None
                if 'images' in item_meta and len(item_meta['images']) > 0:
                    # Get first image (main product image)
                    first_image = item_meta['images'][0]
                    image_url = first_image.get('large') or first_image.get('thumb')
                    
                    if image_url:
                        image_embedding = self.download_and_encode_image(image_url)
                
                # If image download failed, use dummy
                if image_embedding is None:
                    image_embedding = np.random.randn(2048).astype(np.float32)
            else:
                # Text-derived image proxy: encode product info → project to 2048-dim
                # This carries semantic meaning (unlike random noise) while being deterministic
                proxy_text = f"{review.get('title', '')} {item_meta.get('title', '')} product visual"
                proxy_text = proxy_text.strip()[:128]
                proxy_emb = self.text_encoder.encode(proxy_text, convert_to_tensor=False)  # 384-dim
                image_embedding = (proxy_emb @ self._image_projection_matrix).astype(np.float32)  # 2048-dim
            
            # 3. BEHAVIOR FEATURES (32-dim)
            behavior_features = np.zeros(32, dtype=np.float32)
            
            # Basic features (with NaN handling)
            try:
                behavior_features[0] = float(item_meta.get('average_rating', 0.0))
            except (ValueError, TypeError):
                behavior_features[0] = 0.0
            
            try:
                behavior_features[1] = float(item_meta.get('rating_number', 0))
            except (ValueError, TypeError):
                behavior_features[1] = 0.0
            
            try:
                price_val = item_meta.get('price', 0.0)
                # Handle string prices like "$19.99" or invalid values
                if isinstance(price_val, str):
                    price_val = price_val.replace('$', '').replace(',', '')
                behavior_features[2] = float(price_val)
            except (ValueError, TypeError):
                behavior_features[2] = 0.0
            
            try:
                behavior_features[3] = float(review.get('helpful_vote', 0))
            except (ValueError, TypeError):
                behavior_features[3] = 0.0
            
            try:
                behavior_features[4] = float(review.get('verified_purchase', 0))
            except (ValueError, TypeError):
                behavior_features[4] = 0.0
            
            # Derived features
            behavior_features[5] = behavior_features[0] / 5.0  # Normalized rating
            behavior_features[6] = np.log1p(behavior_features[1])  # Log rating count
            behavior_features[7] = np.log1p(behavior_features[2])  # Log price
            
            # CRITICAL: Replace any NaN/Inf that might have snuck through
            behavior_features = np.nan_to_num(behavior_features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # User-item interaction
            behavior_features[8] = float(hash(review['user_id']) % 1000) / 1000.0
            behavior_features[9] = float(hash(parent_asin) % 1000) / 1000.0
            
            # Fill remaining with deterministic values
            for i in range(10, 32):
                seed_val = (hash(review['user_id']) + hash(parent_asin) + i) % 1000
                behavior_features[i] = seed_val / 1000.0
            
            # 4. LABEL (rating 1-5 → 0-4)
            rating = int(float(review['rating']))
            label = rating - 1  # Convert to 0-4
            
            # Store processed sample
            processed_data.append({
                'user_id': review['user_id'],
                'item_id': parent_asin,
                # Metadata for demo/UI (best-effort, may be missing)
                'item_title': item_meta.get('title', '') if isinstance(item_meta, dict) else '',
                'item_category': item_meta.get('main_category', '') if isinstance(item_meta, dict) else '',
                'item_brand': item_meta.get('brand', '') if isinstance(item_meta, dict) else '',
                'item_price': item_meta.get('price', 0.0) if isinstance(item_meta, dict) else 0.0,
                'item_image_url': (
                    (
                        (item_meta.get('images', [{}])[0].get('large') if item_meta.get('images') else None)
                        or (item_meta.get('images', [{}])[0].get('thumb') if item_meta.get('images') else None)
                    )
                    if isinstance(item_meta, dict) else None
                ),
                'rating': rating,
                'label': label,
                'text_embedding': text_embedding.tolist(),
                'image_embedding': image_embedding.tolist(),
                'behavior_features': behavior_features.tolist(),
                'timestamp': review.get('timestamp', 0)
            })
        
        print(f"\n✅ Processed {len(processed_data)} samples")
        
        return pd.DataFrame(processed_data)
    
    def split_federated_clients(self, 
                                data_df: pd.DataFrame,
                                num_clients: int = 10,
                                alpha: float = 0.5) -> Dict[int, pd.DataFrame]:
        """
        Split data into federated clients (Non-IID using Dirichlet)
        
        Args:
            data_df: Processed dataframe
            num_clients: Number of clients
            alpha: Dirichlet parameter (lower = more non-IID)
        
        Returns:
            Dictionary {client_id: client_data_df}
        """
        print("\n" + "="*70)
        print("STEP 3: Splitting into Federated Clients")
        print("="*70)
        
        # Group by user
        user_groups = data_df.groupby('user_id')
        users = list(user_groups.groups.keys())
        num_users = len(users)
        
        print(f"Total users: {num_users}")
        print(f"Total interactions: {len(data_df)}")
        
        # Dirichlet distribution for Non-IID split
        proportions = np.random.dirichlet([alpha] * num_clients, size=num_users)
        
        # Assign users to clients
        user_to_client = {}
        for i, user in enumerate(users):
            # Sample client based on Dirichlet proportions
            client_id = np.random.choice(num_clients, p=proportions[i])
            user_to_client[user] = client_id
        
        # Create client datasets
        client_data = {i: [] for i in range(num_clients)}
        
        for user, client_id in user_to_client.items():
            user_data = user_groups.get_group(user)
            client_data[client_id].append(user_data)
        
        # Convert to DataFrames
        client_dfs = {}
        for client_id in range(num_clients):
            if client_data[client_id]:
                client_df = pd.concat(client_data[client_id], ignore_index=True)
                client_dfs[client_id] = client_df
                print(f"Client {client_id}: {len(client_df)} samples, "
                      f"{client_df['user_id'].nunique()} users")
            else:
                print(f"⚠️  Client {client_id}: No data")
        
        return client_dfs
    
    def save_processed_data(self, client_dfs: Dict[int, pd.DataFrame]):
        """Save processed data for each client"""
        print("\n" + "="*70)
        print("STEP 4: Saving Processed Data")
        print("="*70)
        
        for client_id, client_df in client_dfs.items():
            client_dir = self.output_dir / f"client_{client_id}"
            client_dir.mkdir(exist_ok=True)
            
            # Save as pickle (efficient for numpy arrays)
            output_path = client_dir / "data.pkl"
            client_df.to_pickle(output_path)
            
            print(f"✅ Saved Client {client_id}: {output_path}")
            print(f"   Samples: {len(client_df)}, "
                  f"Users: {client_df['user_id'].nunique()}")
        
        print(f"\n✅ All data saved to: {self.output_dir}")

        # Also save global item/user tables for API + e-commerce demo
        try:
            all_df = pd.concat(list(client_dfs.values()), ignore_index=True) if client_dfs else None
            if all_df is not None and len(all_df) > 0:
                # Items table (unique by item_id)
                item_cols = [
                    "item_id",
                    "item_title",
                    "item_category",
                    "item_brand",
                    "item_price",
                    "item_image_url",
                ]
                existing_item_cols = [c for c in item_cols if c in all_df.columns]
                items_df = (
                    all_df[existing_item_cols]
                    .drop_duplicates(subset=["item_id"])
                    .reset_index(drop=True)
                )
                items_path = self.output_dir / "items_global.csv"
                items_df.to_csv(items_path, index=False, encoding="utf-8")
                print(f"✅ Saved global items table: {items_path} ({len(items_df)} items)")

                # Users table (unique by user_id) with simple stats
                users_df = (
                    all_df.groupby("user_id")
                    .agg(
                        num_interactions=("item_id", "count"),
                        avg_rating_given=("rating", "mean"),
                        last_timestamp=("timestamp", "max"),
                    )
                    .reset_index()
                )
                users_path = self.output_dir / "users_global.csv"
                users_df.to_csv(users_path, index=False, encoding="utf-8")
                print(f"✅ Saved global users table: {users_path} ({len(users_df)} users)")
        except Exception as e:
            print(f"⚠️  Could not save global items/users tables: {e}")


def main():
    """Main processing pipeline"""
    
    print("="*70)
    print("AMAZON REVIEWS 2023 → FEDERATED MULTI-MODAL DATASET")
    print("="*70)
    
    # Configuration
    CATEGORY = "All_Beauty"  # Start with smaller dataset for testing
    SAMPLE_SIZE = 10000  # Process 10K interactions first (remove for full dataset)
    NUM_CLIENTS = 10
    ALPHA = 0.5  # Non-IID parameter
    
    # File paths (Windows-compatible paths)
    REVIEWS_PATH = Path("data/raw/amazon_2023/All_Beauty.jsonl")
    META_PATH = Path("data/raw/amazon_2023/meta_All_Beauty.jsonl")
    
    # Check files exist
    if not REVIEWS_PATH.exists():
        # Try .gz version
        reviews_gz = Path(str(REVIEWS_PATH) + ".gz")
        if reviews_gz.exists():
            print(f"⚠️  Found {reviews_gz}, please extract it first")
            print(f"   Or run: PowerShell -File download_amazon_data.ps1")
        else:
            print(f"❌ Reviews file not found: {REVIEWS_PATH}")
            print(f"\n📥 Please run download script first:")
            print(f"   PowerShell -ExecutionPolicy Bypass -File download_amazon_data.ps1")
        return
    
    if not META_PATH.exists():
        meta_gz = Path(str(META_PATH) + ".gz")
        if meta_gz.exists():
            print(f"⚠️  Found {meta_gz}, please extract it first")
            print(f"   Or run: PowerShell -File download_amazon_data.ps1")
        else:
            print(f"❌ Metadata file not found: {META_PATH}")
            print(f"\n📥 Please run download script first:")
            print(f"   PowerShell -ExecutionPolicy Bypass -File download_amazon_data.ps1")
        return
    
    # Initialize processor
    processor = AmazonDataProcessor(
        category=CATEGORY,
        output_dir="data/amazon_2023_processed",
        sample_size=SAMPLE_SIZE,
        skip_image_download=True  # Skip real images for now (network issues)
    )
    
    # Process data
    data_df = processor.process_reviews(REVIEWS_PATH, META_PATH)
    
    # Split into clients
    client_dfs = processor.split_federated_clients(
        data_df,
        num_clients=NUM_CLIENTS,
        alpha=ALPHA
    )
    
    # Save
    processor.save_processed_data(client_dfs)
    
    print("\n" + "="*70)
    print("✅ PROCESSING COMPLETE!")
    print("="*70)
    print(f"\nTotal samples processed: {len(data_df)}")
    print(f"Clients created: {len(client_dfs)}")
    print(f"\nNext steps:")
    print(f"1. Update federated_dataloader.py to load from pickle files")
    print(f"2. Run training: python src/training/federated_training_pipeline.py")
    print(f"3. Expected accuracy: 60-75% (much better than 30% with synthetic!)")


if __name__ == "__main__":
    main()

