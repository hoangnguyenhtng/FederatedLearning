"""
Milvus Manager for Item Embeddings - FIXED VERSION
Handles storage and retrieval of multi-modal item embeddings
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MilvusManager:
    """Manager for Milvus vector database operations"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        collection_name: str = "item_embeddings",
        embedding_dim: int = 384
    ):
        """
        Initialize Milvus manager
        
        Args:
            host: Milvus server host
            port: Milvus server port
            collection_name: Name of collection
            embedding_dim: Dimension of embeddings
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.collection = None
        
        # Connect to Milvus
        self._connect()
    
    def _connect(self):
        """Connect to Milvus server"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            logger.info(f"✅ Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Milvus: {e}")
            raise
    
    def create_collection(self, drop_existing: bool = False):
        """
        Create collection for item embeddings
        
        Args:
            drop_existing: Drop existing collection if exists
        """
        # Check if collection exists
        if utility.has_collection(self.collection_name):
            if drop_existing:
                logger.info(f"🗑️  Dropping existing collection: {self.collection_name}")
                utility.drop_collection(self.collection_name)
            else:
                logger.info(f"✅ Collection already exists: {self.collection_name}")
                self.collection = Collection(self.collection_name)
                return
        
        # Define schema
        logger.info(f"🔨 Creating collection: {self.collection_name}")
        
        fields = [
            FieldSchema(name="item_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="avg_rating", dtype=DataType.FLOAT),
            FieldSchema(name="num_ratings", dtype=DataType.INT64),
            FieldSchema(name="popularity_score", dtype=DataType.FLOAT),
            FieldSchema(name="price", dtype=DataType.FLOAT),
            FieldSchema(name="brand", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Item embeddings with metadata"
        )
        
        # Create collection
        self.collection = Collection(
            name=self.collection_name,
            schema=schema
        )
        
        logger.info(f"✅ Collection created: {self.collection_name}")
    
    def create_index(self, index_type: str = "HNSW", metric_type: str = "L2"):
        """
        Create index for fast similarity search
        
        Args:
            index_type: Index type (HNSW, IVF_FLAT, etc.)
            metric_type: Distance metric (L2, IP, COSINE)
        """
        logger.info(f"🔨 Creating index: {index_type} with {metric_type}")
        
        index_params = {
            "metric_type": metric_type,
            "index_type": index_type,
            "params": {}
        }
        
        if index_type == "HNSW":
            index_params["params"] = {
                "M": 16,
                "efConstruction": 256
            }
        elif index_type == "IVF_FLAT":
            index_params["params"] = {
                "nlist": 128
            }
        
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        logger.info("✅ Index created successfully")
    
    def insert_items(
        self,
        items_df: pd.DataFrame,
        embeddings: np.ndarray
    ):
        """
        Insert items with embeddings
        
        Args:
            items_df: DataFrame with item metadata
            embeddings: Array of embeddings (N, embedding_dim)
        """
        logger.info(f"📥 Inserting {len(items_df)} items...")
        
        # Validate
        assert len(items_df) == len(embeddings), "Mismatch between items and embeddings"
        assert embeddings.shape[1] == self.embedding_dim, f"Expected dim {self.embedding_dim}, got {embeddings.shape[1]}"
        
        # Prepare data
        data = [
            items_df['item_id'].tolist(),
            items_df['category'].tolist(),
            items_df['name'].tolist(),
            items_df['avg_rating'].tolist(),
            items_df['num_ratings'].tolist(),
            items_df['popularity_score'].tolist(),
            items_df['price'].tolist(),
            items_df['brand'].tolist(),
            embeddings.tolist()
        ]
        
        # Insert
        self.collection.insert(data)
        self.collection.flush()
        
        logger.info(f"✅ Inserted {len(items_df)} items")
    
    def search_similar_items(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for similar items
        
        Args:
            query_embedding: Query vector (1, embedding_dim) or (embedding_dim,)
            top_k: Number of results
            filters: Optional filter expression
            
        Returns:
            List of similar items with metadata
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Load collection
        self.collection.load()
        
        # Search parameters
        search_params = {
            "metric_type": "L2",
            "params": {"ef": 128}  # For HNSW
        }
        
        # Perform search
        results = self.collection.search(
            data=query_embedding.tolist(),
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filters,
            output_fields=["item_id", "category", "name", "avg_rating", 
                          "num_ratings", "popularity_score", "price", "brand"]
        )
        
        # Format results
        similar_items = []
        for hits in results:
            for hit in hits:
                similar_items.append({
                    'item_id': hit.entity.get('item_id'),
                    'category': hit.entity.get('category'),
                    'name': hit.entity.get('name'),
                    'avg_rating': hit.entity.get('avg_rating'),
                    'num_ratings': hit.entity.get('num_ratings'),
                    'popularity_score': hit.entity.get('popularity_score'),
                    'price': hit.entity.get('price'),
                    'brand': hit.entity.get('brand'),
                    'distance': hit.distance,
                    'score': 1.0 / (1.0 + hit.distance)  # Convert distance to similarity
                })
        
        return similar_items
    
    def get_item_by_id(self, item_id: int) -> Optional[Dict]:
        """Get item by ID"""
        self.collection.load()
        
        results = self.collection.query(
            expr=f"item_id == {item_id}",
            output_fields=["*"]
        )
        
        return results[0] if results else None
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        self.collection.load()
        
        return {
            'name': self.collection_name,
            'num_entities': self.collection.num_entities,
            'schema': str(self.collection.schema)
        }
    
    def close(self):
        """Close connection"""
        connections.disconnect("default")
        logger.info("👋 Disconnected from Milvus")


def generate_embeddings_for_items(items_df: pd.DataFrame) -> np.ndarray:
    """
    Generate embeddings for items using their features
    This is a PLACEHOLDER - you should replace with actual multi-modal encoder
    
    Args:
        items_df: DataFrame with item features
        
    Returns:
        embeddings: (N, 384) array
    """
    logger.info("🔮 Generating embeddings for items...")
    
    # PLACEHOLDER: Random embeddings
    # TODO: Replace with actual MultiModalEncoder
    num_items = len(items_df)
    embeddings = np.random.randn(num_items, 384).astype(np.float32)
    
    # Normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    logger.info(f"✅ Generated {num_items} embeddings")
    
    return embeddings


def test_milvus_pipeline():
    """Test complete Milvus pipeline"""
    
    print("\n" + "="*70)
    print("TESTING MILVUS PIPELINE")
    print("="*70)
    
    # 1. Load items data
    print("\n📂 Loading items data...")
    items_path = Path("./data/simulated_clients/items_global.csv")
    
    if not items_path.exists():
        print(f"❌ Items file not found: {items_path}")
        print("💡 Run data generation first:")
        print("   python src/data_generation/main_data_generation.py")
        return
    
    items_df = pd.read_csv(items_path)
    
    # Parse list columns
    if isinstance(items_df['text_keywords'].iloc[0], str):
        items_df['text_keywords'] = items_df['text_keywords'].apply(eval)
    if isinstance(items_df['image_features'].iloc[0], str):
        items_df['image_features'] = items_df['image_features'].apply(eval)
    
    print(f"✅ Loaded {len(items_df)} items")
    print(f"   Columns: {items_df.columns.tolist()}")
    
    # 2. Generate embeddings
    embeddings = generate_embeddings_for_items(items_df)
    
    # 3. Initialize Milvus
    print("\n🔌 Connecting to Milvus...")
    try:
        manager = MilvusManager(
            host="localhost",
            port="19530",
            collection_name="item_embeddings",
            embedding_dim=384
        )
    except Exception as e:
        print(f"\n❌ Cannot connect to Milvus: {e}")
        print("\n💡 Start Milvus first:")
        print("   docker-compose up -d")
        return
    
    # 4. Create collection
    print("\n🔨 Creating collection...")
    manager.create_collection(drop_existing=True)
    
    # 5. Create index
    print("\n🔨 Creating index...")
    manager.create_index(index_type="HNSW", metric_type="L2")
    
    # 6. Insert items
    print("\n📥 Inserting items...")
    manager.insert_items(items_df, embeddings)
    
    # 7. Test search
    print("\n🔍 Testing search...")
    
    # Search for similar items to item 0
    query_embedding = embeddings[0]
    similar_items = manager.search_similar_items(
        query_embedding=query_embedding,
        top_k=5
    )
    
    print(f"\nTop 5 similar items to item 0:")
    for i, item in enumerate(similar_items, 1):
        print(f"{i}. Item {item['item_id']}: {item['name']}")
        print(f"   Category: {item['category']}, Score: {item['score']:.4f}")
    
    # 8. Get collection stats
    print("\n📊 Collection statistics:")
    stats = manager.get_collection_stats()
    for key, value in stats.items():
        if key != 'schema':
            print(f"   {key}: {value}")
    
    # 9. Close connection
    manager.close()
    
    print("\n" + "="*70)
    print("✅ MILVUS PIPELINE TEST COMPLETED")
    print("="*70)


if __name__ == "__main__":
    test_milvus_pipeline()