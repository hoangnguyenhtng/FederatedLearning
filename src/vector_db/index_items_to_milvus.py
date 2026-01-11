"""
Index Item Embeddings to Milvus - FIXED VERSION
Improvements:
1. Better handling of image features (dict vs array)
2. Auto-detect available model checkpoints
3. Fallback to simple embeddings if model not found
4. More robust error handling
"""

import sys
import os
import torch
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging
import json
from sklearn.feature_extraction.text import TfidfVectorizer

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.vector_db.milvus_manager import MilvusManager
from src.models.multimodal_encoder import MultiModalEncoder
from src.models.recommendation_model import FedPerRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_image_features(feature_str: str, expected_dim: int = 2048) -> np.ndarray:
    """
    Parse image features from various formats
    
    Handles:
    - Dict format: "{'brightness': 0.5, 'contrast': 0.3, ...}"
    - Array format: "[0.1, 0.2, 0.3, ...]"
    - Single value: "0.5"
    """
    try:
        if isinstance(feature_str, (int, float)):
            # Single number - repeat it
            return np.full(expected_dim, float(feature_str), dtype=np.float32)
        
        feature_str = str(feature_str).strip()
        
        # Try to parse as dict (extract values)
        if feature_str.startswith('{'):
            try:
                # Remove incomplete dict string
                feature_str = feature_str.rstrip("'")
                if not feature_str.endswith('}'):
                    feature_str += '}'
                
                # Parse dict
                feature_dict = eval(feature_str)
                
                # Extract numeric values
                values = []
                for v in feature_dict.values():
                    if isinstance(v, (int, float)):
                        values.append(float(v))
                
                if len(values) == 0:
                    raise ValueError("No numeric values in dict")
                
                # Repeat/pad to expected_dim
                if len(values) < expected_dim:
                    # Repeat pattern
                    repeat_times = expected_dim // len(values) + 1
                    values = (values * repeat_times)[:expected_dim]
                else:
                    values = values[:expected_dim]
                
                return np.array(values, dtype=np.float32)
                
            except Exception as e:
                logger.debug(f"Failed to parse as dict: {e}")
        
        # Try to parse as array
        if '[' in feature_str:
            feature_str = feature_str.strip('[]')
            if ',' in feature_str:
                values = [float(x.strip()) for x in feature_str.split(',')]
            else:
                values = [float(x) for x in feature_str.split()]
            
            # Pad or trim
            if len(values) < expected_dim:
                values = values + [0.0] * (expected_dim - len(values))
            else:
                values = values[:expected_dim]
            
            return np.array(values, dtype=np.float32)
        
        # Try as single number
        val = float(feature_str)
        return np.full(expected_dim, val, dtype=np.float32)
        
    except Exception as e:
        logger.debug(f"All parsing failed: {e}")
        # Return random features as last resort
        return np.random.randn(expected_dim).astype(np.float32) * 0.01


def load_and_encode_items(data_dir: Path, config: dict) -> dict:
    """
    Load items data and create embeddings
    Returns: Dict with keys ['item_id', 'text_emb', 'image_emb', 'category', 'popularity']
    """
    # Try different paths
    possible_paths = [
        data_dir / "simulated_clients" / "items_global.csv",
        data_dir / "raw" / "items.csv"
    ]
    
    items_path = None
    for path in possible_paths:
        if path.exists():
            items_path = path
            break
    
    if items_path is None:
        raise FileNotFoundError(f"Items data not found in: {possible_paths}")
    
    logger.info(f"📂 Loading items from: {items_path}")
    items_df = pd.read_csv(items_path)
    
    logger.info(f"📋 Columns: {items_df.columns.tolist()}")
    logger.info(f"✅ Loaded {len(items_df)} items")
    
    # TEXT EMBEDDINGS
    logger.info("📝 Creating text embeddings (TF-IDF)...")
    text_keywords = items_df['text_keywords'].fillna('unknown').tolist()
    vectorizer = TfidfVectorizer(max_features=384)
    text_sparse = vectorizer.fit_transform(text_keywords)
    text_embeddings = text_sparse.toarray().astype(np.float32)
    logger.info(f"   ✓ Text shape: {text_embeddings.shape}")
    
    # IMAGE EMBEDDINGS (with improved parsing)
    logger.info("🖼️  Parsing image features...")
    image_embeddings = []
    failed_count = 0
    
    for idx, feat_str in enumerate(items_df['image_features'].tolist()):
        try:
            parsed = parse_image_features(feat_str, expected_dim=2048)
            image_embeddings.append(parsed)
        except Exception as e:
            failed_count += 1
            if failed_count <= 3:  # Only log first few
                logger.warning(f"   Row {idx}: Using random fallback")
            image_embeddings.append(np.random.randn(2048).astype(np.float32) * 0.01)
    
    if failed_count > 0:
        logger.warning(f"   ⚠️  {failed_count}/{len(items_df)} items used fallback features")
    
    image_embeddings = np.array(image_embeddings, dtype=np.float32)
    logger.info(f"   ✓ Image shape: {image_embeddings.shape}")
    
    # METADATA
    item_ids = items_df['item_id'].tolist()
    categories = items_df['category'].tolist()
    popularities = items_df['popularity_score'].tolist()
    
    return {
        'item_id': item_ids,
        'text_emb': text_embeddings,
        'image_emb': image_embeddings,
        'category': categories,
        'popularity': popularities
    }


def find_model_checkpoint(exp_dir: Path) -> Path:
    """
    Find any available model checkpoint
    Returns: Path to checkpoint or None
    """
    search_dirs = [
        exp_dir / "fedper_multimodal_v1" / "models",
        exp_dir / "models",
        exp_dir
    ]
    
    checkpoint_patterns = [
        "global_model_final.pt",
        "global_model_best.pt",
        "global_model_round_*.pt",
        "federated_model_final.pt",
        "server_model_*.pt",
        "*.pt"
    ]
    
    logger.info("🔍 Searching for model checkpoints...")
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        logger.info(f"   Checking: {search_dir}")
        
        for pattern in checkpoint_patterns:
            matches = list(search_dir.glob(pattern))
            if matches:
                # Return the most recent one
                latest = max(matches, key=lambda p: p.stat().st_mtime)
                logger.info(f"   ✓ Found: {latest.name}")
                return latest
    
    return None


def load_trained_model(config: dict, device: torch.device):
    """Load trained model if available"""
    exp_dir = Path(config['paths']['experiments_dir'])
    model_path = find_model_checkpoint(exp_dir)
    
    if model_path is None:
        logger.warning("⚠️  No trained model found")
        return None
    
    logger.info(f"📦 Loading model from: {model_path}")
    
    # Initialize model
    model = FedPerRecommender(
        text_dim=config['model']['text_embedding_dim'],
        image_dim=config['model']['image_embedding_dim'],
        behavior_dim=config['model']['behavior_embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_classes=config['model']['num_classes'],
        num_shared_layers=config['federated']['shared_layers'],
        num_personal_layers=config['federated']['personal_layers']
    )
    
    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', 
                                       checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.to(device)
        
        logger.info("✅ Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return None


def extract_embeddings_with_model(
    model: torch.nn.Module,
    items_data: dict,
    device: torch.device,
    batch_size: int = 128
) -> np.ndarray:
    """Extract embeddings using trained model"""
    model.eval()
    
    text_emb = torch.tensor(items_data['text_emb'], dtype=torch.float32)
    image_emb = torch.tensor(items_data['image_emb'], dtype=torch.float32)
    n_items = len(items_data['item_id'])
    
    all_embeddings = []
    
    logger.info(f"🔄 Extracting embeddings with model...")
    
    with torch.no_grad():
        for i in range(0, n_items, batch_size):
            end_idx = min(i + batch_size, n_items)
            
            batch_text = text_emb[i:end_idx].to(device)
            batch_image = image_emb[i:end_idx].to(device)
            batch_behavior = torch.zeros(len(batch_text), 32, device=device)
            
            # Get embeddings
            if hasattr(model, 'shared_base'):
                if hasattr(model.shared_base, 'multimodal_encoder'):
                    emb = model.shared_base.multimodal_encoder(
                        batch_text, batch_image, batch_behavior
                    )
                else:
                    concat = torch.cat([batch_text, batch_image, batch_behavior], dim=1)
                    emb = model.shared_base.encoder[0](concat)
            else:
                emb = model(batch_text, batch_image, batch_behavior)
            
            all_embeddings.append(emb.cpu().numpy())
            
            if (i // batch_size) % 10 == 0:
                logger.info(f"   Progress: {end_idx}/{n_items}")
    
    final = np.vstack(all_embeddings)
    logger.info(f"✅ Extracted shape: {final.shape}")
    return final


def create_simple_embeddings(items_data: dict, target_dim: int = 256) -> np.ndarray:
    """
    Create simple concatenated embeddings without trained model
    Fallback when model is not available
    """
    logger.info("🔧 Creating simple embeddings (no trained model)...")
    
    text_emb = items_data['text_emb']
    image_emb = items_data['image_emb']
    
    # Reduce dimensions with PCA-like projection
    from sklearn.decomposition import TruncatedSVD
    
    # Reduce text: 384 -> 128
    if text_emb.shape[1] > 128:
        svd_text = TruncatedSVD(n_components=128, random_state=42)
        text_reduced = svd_text.fit_transform(text_emb)
    else:
        text_reduced = text_emb
    
    # Reduce image: 2048 -> 128
    svd_image = TruncatedSVD(n_components=128, random_state=42)
    image_reduced = svd_image.fit_transform(image_emb)
    
    # Concatenate
    combined = np.hstack([text_reduced, image_reduced])
    
    # Final projection to target_dim
    if combined.shape[1] != target_dim:
        svd_final = TruncatedSVD(n_components=target_dim, random_state=42)
        final_emb = svd_final.fit_transform(combined)
    else:
        final_emb = combined
    
    logger.info(f"✅ Simple embeddings shape: {final_emb.shape}")
    return final_emb.astype(np.float32)


def main():
    """Main indexing pipeline with fallback options"""
    
    # Load config
    config_path = Path("configs/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"💻 Device: {device}")
    
    # STEP 1: Load items
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Load and encode items")
    logger.info("="*60)
    
    data_dir = Path(config['paths']['data_dir'])
    items_data = load_and_encode_items(data_dir, config)
    
    # STEP 2: Try to load model
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Load trained model (optional)")
    logger.info("="*60)
    
    model = load_trained_model(config, device)
    
    # STEP 3: Extract embeddings
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Extract final embeddings")
    logger.info("="*60)
    
    if model is not None:
        final_embeddings = extract_embeddings_with_model(
            model, items_data, device,
            batch_size=config['training']['batch_size']
        )
    else:
        logger.info("📌 Using simple embeddings (model not available)")
        final_embeddings = create_simple_embeddings(
            items_data,
            target_dim=config['model']['hidden_dim']
        )
    
    # STEP 4: Connect to Milvus
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Connect to Milvus")
    logger.info("="*60)
    
    milvus_config = config.get('milvus', {})
    
    try:
        manager = MilvusManager(
            host=milvus_config.get('host', 'localhost'),
            port=milvus_config.get('port', '19530'),
            collection_name=milvus_config.get('collection_name', 'item_embeddings'),
            embedding_dim=config['model']['hidden_dim'],
            index_type=milvus_config.get('index_type', 'HNSW'),
            metric_type=milvus_config.get('metric_type', 'L2')
        )
    except Exception as e:
        logger.error(f"❌ Milvus connection failed: {e}")
        logger.info("\n💡 Start Milvus first:")
        logger.info("   docker-compose up -d")
        logger.info("\nOr install standalone:")
        logger.info("   wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml")
        logger.info("   docker-compose up -d")
        return
    
    # STEP 5: Create collection
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Create collection and index")
    logger.info("="*60)
    
    manager.create_collection(drop_existing=True)
    manager.create_index()
    
    # STEP 6: Insert embeddings
    logger.info("\n" + "="*60)
    logger.info("STEP 6: Insert embeddings")
    logger.info("="*60)
    
    manager.insert_embeddings(
        item_ids=items_data['item_id'],
        embeddings=final_embeddings,
        categories=items_data['category'],
        popularities=items_data['popularity'],
        batch_size=1000
    )
    
    # STEP 7: Verify
    logger.info("\n" + "="*60)
    logger.info("✅ INDEXING COMPLETE!")
    logger.info("="*60)
    
    stats = manager.get_collection_stats()
    logger.info(f"Collection: {stats['name']}")
    logger.info(f"Items indexed: {stats['num_entities']}")
    logger.info(f"Embedding dim: {config['model']['hidden_dim']}")
    logger.info("="*60)
    
    # STEP 8: Test search
    logger.info("\n🔍 Testing similarity search...")
    test_query = final_embeddings[0:1]
    results = manager.search(test_query, top_k=5)
    
    logger.info(f"Query item: {items_data['item_id'][0]}")
    logger.info(f"Top 5 similar:")
    for rank, (item_id, dist) in enumerate(results[0], 1):
        logger.info(f"  {rank}. Item {item_id} (dist: {dist:.4f})")
    
    manager.close()
    
    logger.info("\n✅ Success! Next steps:")
    logger.info("  1. Keep Milvus running: docker-compose up -d")
    logger.info("  2. Continue to BƯỚC 11: FastAPI + Streamlit")
    
    if model is None:
        logger.info("\n⚠️  Recommendation: Train model first for better embeddings:")
        logger.info("     python src/training/federated_training_pipeline.py")


if __name__ == "__main__":
    main()