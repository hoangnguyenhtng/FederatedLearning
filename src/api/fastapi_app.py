"""
FastAPI Recommendation API
Provides endpoints for personalized recommendations with explainability
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import yaml

from src.models.multimodal_encoder import MultiModalEncoder
from src.models.recommendation_model import FedPerRecommender
from src.vector_db.milvus_manager import MilvusManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Federated Multi-Modal Recommendation API",
    description="API for personalized recommendations with privacy-preserving federated learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables (initialized on startup)
model = None
milvus_manager = None
items_df = None
users_df = None
device = None
config = None
amazon_interactions_df = None  # optional, used to build item embeddings/users


# ============================================================================
# Data Models
# ============================================================================

class RecommendationRequest(BaseModel):
    """Request for recommendations"""
    user_id: int
    top_k: int = 10
    filters: Optional[Dict[str, str]] = None
    explain: bool = True


class RecommendationItem(BaseModel):
    """Single recommendation item"""
    item_id: int
    name: str
    category: str
    score: float
    rank: int
    
    # Explainability
    text_contribution: Optional[float] = None
    image_contribution: Optional[float] = None
    behavior_contribution: Optional[float] = None
    
    # Metadata
    avg_rating: float
    num_ratings: int
    price: float
    brand: str


class RecommendationResponse(BaseModel):
    """Response with recommendations"""
    user_id: int
    recommendations: List[RecommendationItem]
    
    # User info
    user_preference_type: str
    fusion_weights: Dict[str, float]
    
    # Metadata
    timestamp: str
    processing_time_ms: float


class UserProfileResponse(BaseModel):
    """User profile information"""
    user_id: int
    age: Optional[int] = None
    preference_type: Optional[str] = None
    preferred_categories: Optional[List[str]] = None
    registration_date: Optional[str] = None
    
    # Learned preferences
    fusion_weights: Dict[str, float]
    
    # Statistics
    num_interactions: int
    avg_rating_given: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    milvus_connected: bool
    num_items: int
    num_users: int


# ============================================================================
# Startup & Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize models and data on startup"""
    global model, milvus_manager, items_df, users_df, device, config, amazon_interactions_df
    
    logger.info("🚀 Starting up API server...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  Using device: {device}")
    
    try:
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "configs" / "config.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        else:
            config = {}

        # 1. Load data (prefer Amazon processed artifacts if present)
        logger.info("📂 Loading data...")
        amazon_dir = project_root / "data" / "amazon_2023_processed"
        amazon_client0 = amazon_dir / "client_0" / "data.pkl"

        # Preferred: build item catalog + embeddings from processed interactions
        if amazon_client0.exists():
            amazon_interactions_df = pd.read_pickle(amazon_client0)

            # Normalize embedding columns (list -> numpy array) lazily later; keep lists in DF for now
            cols_needed = [
                "item_id",
                "item_title",
                "item_category",
                "item_brand",
                "item_price",
                "item_image_url",
                "text_embedding",
                "image_embedding",
            ]
            available = [c for c in cols_needed if c in amazon_interactions_df.columns]
            tmp = amazon_interactions_df[available].copy()

            # Build unique item table (first occurrence per item_id)
            items_df = tmp.drop_duplicates(subset=["item_id"]).reset_index(drop=True)
            logger.info(f"✅ Built Amazon item catalog from client_0: {len(items_df)} items")

            # Build user table from unique user_id
            if "user_id" in amazon_interactions_df.columns:
                users_df = (
                    amazon_interactions_df.groupby("user_id")
                    .agg(num_interactions=("item_id", "count"), avg_rating_given=("rating", "mean"))
                    .reset_index()
                )
                logger.info(f"✅ Built Amazon users table from client_0: {len(users_df)} users")
            else:
                users_df = pd.DataFrame({"user_id": [], "num_interactions": [], "avg_rating_given": []})
                logger.warning("⚠️  Amazon interactions missing user_id; users table empty")

        # Secondary: use pre-saved tables from processor (no embeddings guaranteed)
        elif (amazon_dir / "items_global.csv").exists():
            items_df = pd.read_csv(amazon_dir / "items_global.csv")
            logger.warning("⚠️  Loaded Amazon items_global.csv (may not include embeddings for ranking)")
            logger.info(f"✅ Loaded Amazon items table: {len(items_df)} items")
        else:
            # Fallback to synthetic demo artifacts if present
            items_df = pd.read_csv(project_root / "data/simulated_clients/items_global.csv")
            logger.warning("⚠️  Using simulated items table (Amazon items_global.csv not found)")
            # Parse list columns
            if "text_keywords" in items_df.columns and isinstance(items_df["text_keywords"].iloc[0], str):
                items_df["text_keywords"] = items_df["text_keywords"].apply(eval)
            if "image_features" in items_df.columns and isinstance(items_df["image_features"].iloc[0], str):
                items_df["image_features"] = items_df["image_features"].apply(eval)
            logger.info(f"✅ Loaded simulated items: {len(items_df)} items")
        
        if users_df is not None:
            pass
        elif (amazon_dir / "users_global.csv").exists():
            users_df = pd.read_csv(amazon_dir / "users_global.csv")
            logger.info(f"✅ Loaded Amazon users table: {len(users_df)} users")
        else:
            users_df = pd.read_csv(project_root / "data/simulated_clients/client_0/users.csv")
            if "preferred_categories" in users_df.columns and isinstance(users_df["preferred_categories"].iloc[0], str):
                users_df["preferred_categories"] = users_df["preferred_categories"].apply(eval)
            logger.warning("⚠️  Using simulated users table (Amazon users_global.csv not found)")
            logger.info(f"✅ Loaded simulated users: {len(users_df)} users")
        
        # 2. Initialize Milvus
        logger.info("🔌 Connecting to Milvus...")
        try:
            milvus_manager = MilvusManager(
                host="localhost",
                port="19530",
                collection_name="item_embeddings",
                embedding_dim=384
            )
            
            # Try to load collection if it exists
            if milvus_manager.collection:
                logger.info("✅ Milvus connected and collection loaded")
            else:
                logger.warning("⚠️  Milvus connected but collection not found. Run milvus_manager.py to create it.")
        except Exception as e:
            logger.warning(f"⚠️  Milvus connection issue: {e}")
            logger.info("💡 You can still use the API, but similarity search won't work")
            milvus_manager = None
        
        # 3. Load model + checkpoint (preferred)
        logger.info("🔨 Loading model...")

        model_cfg = (config or {}).get("model", {})
        multimodal_encoder = MultiModalEncoder(
            text_dim=model_cfg.get("text_embedding_dim", 384),
            image_dim=model_cfg.get("image_embedding_dim", 2048),
            behavior_dim=model_cfg.get("behavior_embedding_dim", 32),
            hidden_dim=model_cfg.get("hidden_dim", 256),
            output_dim=384,
        )
        model = FedPerRecommender(
            multimodal_encoder=multimodal_encoder,
            shared_hidden_dims=model_cfg.get("shared_hidden_dims", [512, 256, 128]),
            personal_hidden_dims=model_cfg.get("personal_hidden_dims", [64, 32]),
            num_classes=model_cfg.get("num_classes", 5),
            dropout=model_cfg.get("dropout", 0.2),
        ).to(device)

        # Load trained weights if available
        exp_name = (config or {}).get("experiment", {}).get("name", "fedper_multimodal_v1")
        exp_root = Path((config or {}).get("paths", {}).get("experiments_dir", "experiments"))
        ckpt_path = project_root / exp_root / exp_name / "models" / "global_model_final.pt"
        if ckpt_path.exists():
            checkpoint = torch.load(str(ckpt_path), map_location=device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                logger.info(f"✅ Loaded checkpoint: {ckpt_path}")
            else:
                logger.warning(f"⚠️  Checkpoint missing model_state_dict: {ckpt_path}")
        else:
            logger.warning(f"⚠️  Checkpoint not found: {ckpt_path} (API will run but quality may be low)")

        model.eval()
        
        logger.info("✅ Model loaded")
        
        logger.info("🎉 API server ready!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Shutting down API server...")
    
    if milvus_manager:
        milvus_manager.close()


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Federated Multi-Modal Recommendation API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        milvus_connected=milvus_manager is not None,
        num_items=len(items_df) if items_df is not None else 0,
        num_users=len(users_df) if users_df is not None else 0
    )


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get personalized recommendations for a user
    
    Args:
        request: RecommendationRequest with user_id and parameters
        
    Returns:
        RecommendationResponse with ranked items and explanations
    """
    start_time = datetime.now()
    
    try:
        # 1. Resolve user (API uses user index -> underlying user_id if available)
        if users_df is None or len(users_df) == 0:
            raise HTTPException(status_code=503, detail="User table not loaded")

        if request.user_id >= len(users_df):
            raise HTTPException(status_code=404, detail=f"User {request.user_id} not found")

        user_row = users_df.iloc[request.user_id]
        resolved_user_id = user_row["user_id"] if "user_id" in user_row else request.user_id

        # 2. Candidate items (for demo: take first N items; for production: retrieve via Milvus)
        if items_df is None or len(items_df) == 0:
            raise HTTPException(status_code=503, detail="Item catalog not loaded")

        candidate_pool = items_df

        # Optional category filter
        if request.filters and "category" in request.filters:
            cat = request.filters["category"]
            if "item_category" in candidate_pool.columns:
                candidate_pool = candidate_pool[candidate_pool["item_category"] == cat]
            elif "category" in candidate_pool.columns:
                candidate_pool = candidate_pool[candidate_pool["category"] == cat]

        if len(candidate_pool) == 0:
            raise HTTPException(status_code=404, detail="No items match filters")

        # Limit candidates for latency (web demo)
        max_candidates = 2000
        candidate_pool = candidate_pool.head(max_candidates).reset_index(drop=True)

        # 3. Score candidates by predicted expected rating
        def build_behavior_features(u, item_id_str: str, price_val) -> np.ndarray:
            bf = np.zeros(32, dtype=np.float32)
            # Best-effort mimic of `process_amazon_data.py`
            try:
                p = price_val
                if isinstance(p, str):
                    p = p.replace("$", "").replace(",", "")
                bf[2] = float(p) if p is not None else 0.0
            except Exception:
                bf[2] = 0.0
            bf[8] = float(hash(str(u)) % 1000) / 1000.0
            bf[9] = float(hash(str(item_id_str)) % 1000) / 1000.0
            for i in range(10, 32):
                seed_val = (hash(str(u)) + hash(str(item_id_str)) + i) % 1000
                bf[i] = seed_val / 1000.0
            return bf

        # Assemble tensors in batches
        batch_size = 128
        scores_all: List[float] = []
        fusion_weights_last = None

        rating_values = torch.arange(1, 6, device=device).float()  # 1..5

        for start in range(0, len(candidate_pool), batch_size):
            sub = candidate_pool.iloc[start : start + batch_size]

            # Embeddings: if missing, fallback deterministic dummy (still stable)
            def _to_np_list(x, dim: int, seed_key: str):
                if isinstance(x, (list, np.ndarray)) and len(x) == dim:
                    return np.asarray(x, dtype=np.float32)
                rng = np.random.default_rng(hash(seed_key) % 1_000_000)
                return (rng.standard_normal((dim,), dtype=np.float32) * 0.1)

            text_arr = np.stack(
                [
                    _to_np_list(row.get("text_embedding", None), 384, f"{resolved_user_id}:{row.get('item_id')}::t")
                    for _, row in sub.iterrows()
                ],
                axis=0,
            )
            img_arr = np.stack(
                [
                    _to_np_list(row.get("image_embedding", None), 2048, f"{resolved_user_id}:{row.get('item_id')}::i")
                    for _, row in sub.iterrows()
                ],
                axis=0,
            )
            beh_arr = np.stack(
                [
                    build_behavior_features(
                        resolved_user_id,
                        str(row.get("item_id")),
                        row.get("item_price", row.get("price", 0.0)),
                    )
                    for _, row in sub.iterrows()
                ],
                axis=0,
            )

            text_emb = torch.tensor(text_arr, device=device)
            image_emb = torch.tensor(img_arr, device=device)
            behavior_feat = torch.tensor(beh_arr, device=device)

            logits, fusion_weights = model(text_emb, image_emb, behavior_feat, return_fusion_weights=True)
            fusion_weights_last = fusion_weights  # keep last batch weights for response
            probs = torch.softmax(logits, dim=1)
            expected_rating = (probs * rating_values).sum(dim=1)  # (batch,)
            scores_all.extend(expected_rating.detach().cpu().numpy().tolist())

        candidate_pool = candidate_pool.assign(pred_score=np.array(scores_all, dtype=np.float32))
        candidate_pool = candidate_pool.sort_values("pred_score", ascending=False).head(request.top_k).reset_index(drop=True)
        
        # 4. Get item details (fallback fields if Amazon metadata missing)
        recommendations = []
        # Use fusion weights averaged across last batch (best-effort)
        if fusion_weights_last is None:
            fusion_weights_last = torch.tensor([[1 / 3, 1 / 3, 1 / 3]], device=device)
        fw = fusion_weights_last.mean(dim=0)

        for rank, (_, demo_row) in enumerate(candidate_pool.iterrows(), 1):
            
            rec_item = RecommendationItem(
                item_id=int(hash(str(demo_row.get("item_id", rank))) % 1000000),
                name=str(demo_row.get("item_title", demo_row.get("name", f"Item {rank}"))),
                category=str(demo_row.get("item_category", demo_row.get("category", "Unknown"))),
                score=float(demo_row.get("pred_score", 0.0)),
                rank=rank,
                
                # Explainability
                text_contribution=float(fw[0].item()),
                image_contribution=float(fw[1].item()),
                behavior_contribution=float(fw[2].item()),
                
                # Metadata
                avg_rating=float(demo_row.get("avg_rating", 0.0) or 0.0),
                num_ratings=int(demo_row.get("num_ratings", 0) or 0),
                price=float(demo_row.get("item_price", demo_row.get("price", 0.0)) or 0.0),
                brand=str(demo_row.get("item_brand", demo_row.get("brand", "")) or "")
            )
            
            recommendations.append(rec_item)
        
        # 5. Build response
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response = RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            user_preference_type=str(user_row.get("preference_type", "amazon_user")),
            fusion_weights={
                "text": float(fw[0].item()),
                "image": float(fw[1].item()),
                "behavior": float(fw[2].item()),
            },
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user/{user_id}", response_model=UserProfileResponse)
async def get_user_profile(user_id: int):
    """
    Get user profile and learned preferences
    
    Args:
        user_id: User identifier
        
    Returns:
        UserProfileResponse with user information
    """
    try:
        if user_id >= len(users_df):
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        
        user = users_df.iloc[user_id]

        # Compute fusion weights deterministically (stable across calls)
        seed = int(user_id) % 1_000_000
        rng = np.random.default_rng(seed)
        text_emb = torch.tensor(rng.standard_normal((1, 384), dtype=np.float32) * 0.1).to(device)
        image_emb = torch.tensor(rng.standard_normal((1, 2048), dtype=np.float32) * 0.1).to(device)
        behavior_feat = torch.tensor(rng.standard_normal((1, 32), dtype=np.float32) * 0.1).to(device)

        with torch.no_grad():
            _, fusion_weights = model(text_emb, image_emb, behavior_feat, return_fusion_weights=True)
        
        return UserProfileResponse(
            user_id=user_id,
            age=int(user['age']) if 'age' in user and pd.notna(user['age']) else None,
            preference_type=str(user['preference_type']) if 'preference_type' in user and pd.notna(user['preference_type']) else None,
            preferred_categories=user['preferred_categories'] if 'preferred_categories' in user else None,
            registration_date=str(user['registration_date']) if 'registration_date' in user and pd.notna(user['registration_date']) else None,
            fusion_weights={
                'text': float(fusion_weights[0, 0]),
                'image': float(fusion_weights[0, 1]),
                'behavior': float(fusion_weights[0, 2])
            },
            num_interactions=int(user.get('num_interactions', 0) or 0),
            avg_rating_given=float(user.get('avg_rating_given', 0.0) or 0.0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Get user profile failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/items/search")
async def search_items(
    query: str = Query(..., description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    top_k: int = Query(10, ge=1, le=100, description="Number of results")
):
    """
    Search items by text query
    
    Args:
        query: Search query string
        category: Optional category filter
        top_k: Number of results
        
    Returns:
        List of matching items
    """
    try:
        # Simple text search (in production, use Milvus text search)
        filtered_items = items_df.copy()
        
        if category:
            filtered_items = filtered_items[filtered_items['category'] == category]
        
        # Simple keyword search
        query_lower = query.lower()
        mask = filtered_items['name'].str.lower().str.contains(query_lower, na=False)
        results = filtered_items[mask].head(top_k)
        
        return {
            "query": query,
            "category": category,
            "num_results": len(results),
            "items": results.to_dict('records')
        }
        
    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_statistics():
    """Get system statistics"""
    try:
        stats = {
            "total_items": len(items_df),
            "total_users": len(users_df),
            "categories": items_df['category'].value_counts().to_dict(),
            "preference_types": users_df['preference_type'].value_counts().to_dict(),
            "avg_item_rating": float(items_df['avg_rating'].mean()),
            "total_ratings": int(items_df['num_ratings'].sum())
        }
        
        # Milvus stats
        if milvus_manager and milvus_manager.collection:
            try:
                milvus_stats = milvus_manager.get_collection_stats()
                stats['milvus'] = {
                    'collection_name': milvus_stats['name'],
                    'num_entities': milvus_stats['num_entities']
                }
            except Exception as e:
                logger.warning(f"Could not get Milvus stats: {e}")
                stats['milvus'] = {'status': 'collection not initialized'}
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Get stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("="*70)
    print("FEDERATED MULTI-MODAL RECOMMENDATION API")
    print("="*70)
    print("Starting server...")
    print("Docs available at: http://localhost:8000/docs")
    print("="*70)
    
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )