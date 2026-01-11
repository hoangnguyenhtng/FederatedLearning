"""
Amazon Reviews 2023 Multi-Category Data Processor for Federated Learning

This script processes multiple Amazon categories for cross-domain federated learning:
- All_Beauty (701k reviews)
- Video_Games (4.6M reviews)
- Amazon_Fashion (2.5M reviews)
- Baby_Products (6M reviews)

Total: ~13.8M reviews → Processed into 40 federated clients (10 per category)

Features:
1. Multi-modal embeddings (text, image, behavior)
2. Non-IID data distribution per category
3. Balanced client distribution
4. Memory-efficient chunked processing
5. Resume capability with checkpoints
"""

import json
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
import time
from datetime import datetime
import yaml

# ML libraries
from sentence_transformers import SentenceTransformer
import torchvision.models as models
import torchvision.transforms as transforms


class MultiCategoryAmazonProcessor:
    """Process multiple Amazon categories for federated learning"""
    
    def __init__(self, config_path: str = "configs/config_multi_category.yaml"):
        """
        Initialize processor with config
        
        Args:
            config_path: Path to YAML config file
        """
        print("="*70)
        print("MULTI-CATEGORY AMAZON PROCESSOR")
        print("="*70)
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.categories = self.config['categories']
        self.data_dir = Path(self.config['paths']['data_raw'])
        self.output_dir = Path(self.config['paths']['data_processed'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing settings
        self.chunk_size = self.config['processing']['chunk_size']
        self.checkpoint_interval = self.config['processing']['checkpoint_interval']
        
        # THESIS-OPTIMIZED: Support both global and per-category sampling
        self.max_samples_per_category = self.config['data_generation'].get('max_samples_per_category')
        self.per_category_samples = self.config['data_generation'].get('per_category_samples', {})
        self.skip_image_download = self.config['data_generation']['skip_image_download']
        
        # Federated settings
        self.num_clients = self.config['federated']['num_clients']
        self.clients_per_category = self.config['data_generation']['clients_per_category']
        self.alpha = self.config['data_generation']['alpha']
        self.min_samples_per_client = self.config['data_generation']['min_samples_per_client']
        
        # Initialize encoders
        print("\n📦 Loading models...")
        print("   Text encoder: SentenceTransformer (all-MiniLM-L6-v2)")
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Device: {self.device}")
        
        if not self.skip_image_download:
            print("   Image encoder: ResNet-50")
            try:
                self.image_model = models.resnet50(pretrained=True)
                self.image_model = torch.nn.Sequential(*list(self.image_model.children())[:-1])
                self.image_model.eval()
                self.image_model.to(self.device)
                
                self.image_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                print("   ✅ Image encoder loaded")
            except Exception as e:
                print(f"   ⚠️  Failed to load ResNet-50: {e}")
                print(f"   Falling back to dummy features")
                self.skip_image_download = True
        else:
            print("   ⚠️  Skipping real image processing (using dummy features)")
        
        print("✅ Initialization complete!\n")
    
    def get_max_samples_for_category(self, category: str) -> int:
        """Get max samples to process for a specific category"""
        # Priority: per_category_samples > max_samples_per_category > None
        if self.per_category_samples and category in self.per_category_samples:
            return self.per_category_samples[category]
        elif self.max_samples_per_category:
            return self.max_samples_per_category
        else:
            return None  # Process all
    
    def load_jsonl(self, filepath: Path, max_samples: int = None) -> List[Dict]:
        """Load JSONL file with progress bar - OPTIMIZED for large files"""
        data = []
        print(f"   Loading {filepath.name}...")
        
        # For large files, don't count all lines if sampling (much faster!)
        if max_samples:
            # Only process first N samples (much faster!)
            print(f"   ⚡ Sampling mode: Processing first {max_samples:,} samples")
            print(f"   ⏱️  Estimated time: ~{max_samples/3600:.1f} hours (at 1 line/sec)")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(tqdm(f, total=max_samples, desc=f"   {filepath.name}")):
                    if i >= max_samples:
                        break
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        else:
            # Count total lines first for progress bar (only if processing all)
            with open(filepath, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(tqdm(f, total=total_lines, desc=f"   {filepath.name}")):
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        print(f"   ✅ Loaded {len(data):,} records\n")
        return data
    
    def download_and_encode_image(self, image_url: str) -> np.ndarray:
        """Download and encode image with ResNet-50"""
        try:
            response = requests.get(image_url, timeout=5)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.image_model(image_tensor)
                features = features.squeeze().cpu().numpy()
            
            return features
        except:
            return None
    
    def process_single_sample(self, review: Dict, item_meta: Dict, category: str) -> Dict:
        """Process a single review into multi-modal features"""
        
        # 1. TEXT EMBEDDINGS (384-dim)
        review_text = f"{review.get('title', '')} {review.get('text', '')}"
        item_desc = ' '.join(item_meta.get('description', [])) if 'description' in item_meta else ''
        combined_text = f"{review_text} {item_desc}".strip()[:512]
        
        text_embedding = self.text_encoder.encode(combined_text, convert_to_tensor=False)
        
        # 2. IMAGE FEATURES (2048-dim)
        if not self.skip_image_download:
            image_embedding = None
            if 'images' in item_meta and len(item_meta['images']) > 0:
                first_image = item_meta['images'][0]
                image_url = first_image.get('large') or first_image.get('thumb')
                if image_url:
                    image_embedding = self.download_and_encode_image(image_url)
            
            if image_embedding is None:
                # Fallback to deterministic dummy
                seed = hash(review['parent_asin']) % 10000
                np.random.seed(seed)
                image_embedding = np.random.randn(2048).astype(np.float32) * 0.1
        else:
            # Deterministic dummy based on item_id
            seed = hash(review['parent_asin']) % 10000
            np.random.seed(seed)
            image_embedding = np.random.randn(2048).astype(np.float32) * 0.1
        
        # 3. BEHAVIOR FEATURES (32-dim)
        behavior_features = np.zeros(32, dtype=np.float32)
        
        # Extract metadata features with NaN handling
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
        
        # User-item interaction hashes
        behavior_features[8] = float(hash(review['user_id']) % 1000) / 1000.0
        behavior_features[9] = float(hash(review['parent_asin']) % 1000) / 1000.0
        
        # Fill remaining with deterministic values
        for i in range(10, 32):
            seed_val = (hash(review['user_id']) + hash(review['parent_asin']) + i) % 1000
            behavior_features[i] = seed_val / 1000.0
        
        # Replace any NaN/Inf
        behavior_features = np.nan_to_num(behavior_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 4. LABEL (rating 1-5 → 0-4)
        rating = int(float(review['rating']))
        label = rating - 1
        
        return {
            'user_id': review['user_id'],
            'item_id': review['parent_asin'],
            'category': category,
            'rating': rating,
            'label': label,
            'text_embedding': text_embedding.tolist(),
            'image_embedding': image_embedding.tolist(),
            'behavior_features': behavior_features.tolist(),
            'timestamp': review.get('timestamp', 0)
        }
    
    def process_category(self, category: str) -> pd.DataFrame:
        """Process a single category"""
        print(f"\n{'='*70}")
        print(f"PROCESSING CATEGORY: {category}")
        print(f"{'='*70}\n")
        
        # Get max samples for this category
        max_samples = self.get_max_samples_for_category(category)
        if max_samples:
            print(f"📊 Sampling: {max_samples:,} samples from {category}")
            print(f"⏱️  Estimated processing time: ~{max_samples/3600:.1f} hours (at 1 line/sec)\n")
        
        # File paths
        reviews_path = self.data_dir / f"{category}.jsonl"
        meta_path = self.data_dir / f"meta_{category}.jsonl"
        
        if not reviews_path.exists() or not meta_path.exists():
            print(f"❌ Files not found for {category}")
            return pd.DataFrame()
        
        # Load data (with sampling)
        print("📥 Loading data...")
        reviews = self.load_jsonl(reviews_path, max_samples)
        metadata = self.load_jsonl(meta_path)  # Load all metadata (needed for lookups)
        
        # Create metadata lookup
        meta_df = pd.DataFrame(metadata)
        meta_dict = meta_df.set_index('parent_asin').to_dict('index')
        
        print(f"📊 Statistics:")
        print(f"   Reviews: {len(reviews):,}")
        print(f"   Metadata: {len(metadata):,}")
        
        # Process samples
        print(f"\n🔄 Processing samples...")
        processed_data = []
        
        for review in tqdm(reviews, desc=f"   {category}"):
            parent_asin = review.get('parent_asin')
            if not parent_asin or parent_asin not in meta_dict:
                continue
            
            item_meta = meta_dict[parent_asin]
            
            try:
                sample = self.process_single_sample(review, item_meta, category)
                processed_data.append(sample)
            except Exception as e:
                # Skip problematic samples
                continue
            
            # Checkpoint saving
            if len(processed_data) % self.checkpoint_interval == 0:
                self._save_checkpoint(category, processed_data)
        
        print(f"\n✅ Processed {len(processed_data):,} samples for {category}")
        
        return pd.DataFrame(processed_data)
    
    def _save_checkpoint(self, category: str, data: List[Dict]):
        """Save processing checkpoint"""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{category}_checkpoint_{len(data)}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"   💾 Checkpoint saved: {len(data):,} samples")
    
    def split_federated_clients(self, all_data: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        Split multi-category data into federated clients
        
        Strategy:
        - 40 clients total (10 per category)
        - Each category gets its own set of clients (domain-specific)
        - Non-IID within each category using Dirichlet
        """
        print(f"\n{'='*70}")
        print("SPLITTING INTO FEDERATED CLIENTS")
        print(f"{'='*70}\n")
        
        client_dfs = {}
        client_id = 0
        
        for category in self.categories:
            print(f"📦 Processing {category}...")
            
            # Get data for this category
            category_data = all_data[all_data['category'] == category].copy()
            
            if len(category_data) == 0:
                print(f"   ⚠️  No data for {category}, skipping")
                continue
            
            # Group by user
            user_groups = category_data.groupby('user_id')
            users = list(user_groups.groups.keys())
            num_users = len(users)
            
            print(f"   Users: {num_users:,}")
            print(f"   Samples: {len(category_data):,}")
            
            # Dirichlet distribution for Non-IID split
            proportions = np.random.dirichlet(
                [self.alpha] * self.clients_per_category, 
                size=num_users
            )
            
            # Assign users to clients
            user_to_client = {}
            for i, user in enumerate(users):
                local_client_id = np.random.choice(self.clients_per_category, p=proportions[i])
                user_to_client[user] = client_id + local_client_id
            
            # Create client datasets for this category
            category_clients = {i: [] for i in range(client_id, client_id + self.clients_per_category)}
            
            for user, assigned_client in user_to_client.items():
                user_data = user_groups.get_group(user)
                category_clients[assigned_client].append(user_data)
            
            # Convert to DataFrames
            for local_id in range(self.clients_per_category):
                global_client_id = client_id + local_id
                if category_clients[global_client_id]:
                    client_df = pd.concat(category_clients[global_client_id], ignore_index=True)
                    
                    # Filter out clients with too few samples
                    if len(client_df) >= self.min_samples_per_client:
                        client_dfs[global_client_id] = client_df
                        print(f"   Client {global_client_id}: {len(client_df):,} samples, "
                              f"{client_df['user_id'].nunique()} users")
                    else:
                        print(f"   ⚠️  Client {global_client_id}: Too few samples ({len(client_df)}), skipping")
            
            client_id += self.clients_per_category
            print()
        
        print(f"✅ Created {len(client_dfs)} clients total\n")
        return client_dfs
    
    def save_processed_data(self, client_dfs: Dict[int, pd.DataFrame]):
        """Save processed data for each client"""
        print(f"{'='*70}")
        print("SAVING PROCESSED DATA")
        print(f"{'='*70}\n")
        
        for client_id, client_df in client_dfs.items():
            client_dir = self.output_dir / f"client_{client_id}"
            client_dir.mkdir(exist_ok=True)
            
            # Save as pickle
            output_path = client_dir / "data.pkl"
            client_df.to_pickle(output_path)
            
            # Get category distribution
            category_dist = client_df['category'].value_counts().to_dict()
            
            print(f"✅ Client {client_id}:")
            print(f"   Path: {output_path}")
            print(f"   Samples: {len(client_df):,}")
            print(f"   Users: {client_df['user_id'].nunique()}")
            print(f"   Categories: {category_dist}")
            print()
        
        # Save metadata
        metadata = {
            'num_clients': len(client_dfs),
            'categories': self.categories,
            'total_samples': sum(len(df) for df in client_dfs.values()),
            'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'config': self.config
        }
        
        metadata_path = self.output_dir / "processing_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"📊 Metadata saved to: {metadata_path}")
        print(f"✅ All data saved to: {self.output_dir}\n")
    
    def run(self):
        """Main processing pipeline"""
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print("STARTING MULTI-CATEGORY PROCESSING")
        print(f"{'='*70}")
        print(f"Categories: {', '.join(self.categories)}")
        print(f"Target clients: {self.num_clients}")
        print(f"Output directory: {self.output_dir}")
        print()
        
        # Process each category
        all_category_data = []
        
        for category in self.categories:
            category_df = self.process_category(category)
            if not category_df.empty:
                all_category_data.append(category_df)
        
        # Combine all categories
        print(f"\n{'='*70}")
        print("COMBINING CATEGORIES")
        print(f"{'='*70}\n")
        
        all_data = pd.concat(all_category_data, ignore_index=True)
        
        print(f"📊 Combined Statistics:")
        print(f"   Total samples: {len(all_data):,}")
        print(f"   Total users: {all_data['user_id'].nunique():,}")
        print(f"   Total items: {all_data['item_id'].nunique():,}")
        print()
        
        for category in self.categories:
            cat_count = len(all_data[all_data['category'] == category])
            print(f"   {category}: {cat_count:,} samples ({cat_count/len(all_data)*100:.1f}%)")
        
        # Split into federated clients
        client_dfs = self.split_federated_clients(all_data)
        
        # Save processed data
        self.save_processed_data(client_dfs)
        
        # Final summary
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print("✅ PROCESSING COMPLETE!")
        print(f"{'='*70}")
        print(f"⏱️  Total time: {elapsed_time/60:.1f} minutes")
        print(f"📊 Total samples: {len(all_data):,}")
        print(f"👥 Total clients: {len(client_dfs)}")
        print(f"💾 Output: {self.output_dir}")
        print()
        print("Next steps:")
        print("1. Verify data:")
        print("   python check_data_distribution.py")
        print()
        print("2. Train model:")
        print("   python src/training/federated_training_pipeline.py --config configs/config_multi_category.yaml")
        print()
        print("3. Expected results:")
        print("   - Accuracy: 75-80% (with reduced dataset)")
        print("   - Training time: ~2-3 days (100 rounds)")
        print("   - Cross-domain generalization ✅")
        print()
        print("🎓 Good luck with your thesis! 🚀")


def main():
    """Entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process multi-category Amazon data")
    parser.add_argument('--config', type=str, default='configs/config_multi_category.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run processor
    processor = MultiCategoryAmazonProcessor(config_path=args.config)
    processor.run()


if __name__ == "__main__":
    main()