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
    age: int
    preference_type: str
    preferred_categories: List[str]
    registration_date: str
    
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
    global model, milvus_manager, items_df, users_df, device
    
    logger.info("🚀 Starting up API server...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  Using device: {device}")
    
    try:
        # 1. Load data
        logger.info("📂 Loading data...")
        project_root = Path(__file__).parent.parent.parent
        
        items_df = pd.read_csv(project_root / "data/simulated_clients/items_global.csv")
        
        # Parse list columns
        if isinstance(items_df['text_keywords'].iloc[0], str):
            items_df['text_keywords'] = items_df['text_keywords'].apply(eval)
        if isinstance(items_df['image_features'].iloc[0], str):
            items_df['image_features'] = items_df['image_features'].apply(eval)
        
        logger.info(f"✅ Loaded {len(items_df)} items")
        
        # Load user data (from client 0 as example)
        users_df = pd.read_csv(project_root / "data/simulated_clients/client_0/users.csv")
        if isinstance(users_df['preferred_categories'].iloc[0], str):
            users_df['preferred_categories'] = users_df['preferred_categories'].apply(eval)
        
        logger.info(f"✅ Loaded {len(users_df)} users")
        
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
        
        # 3. Load model (placeholder - replace with trained model)
        logger.info("🔨 Loading model...")
        multimodal_encoder = MultiModalEncoder()
        model = FedPerRecommender(
            multimodal_encoder=multimodal_encoder,
            num_items=len(items_df)
        )
        model.to(device)
        model.eval()
        
        # TODO: Load trained weights
        # checkpoint = torch.load("experiments/.../global_model_final.pt")
        # model.load_state_dict(checkpoint['model_state_dict'])
        
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
        # 1. Get user profile
        if request.user_id >= len(users_df):
            raise HTTPException(status_code=404, detail=f"User {request.user_id} not found")
        
        user = users_df.iloc[request.user_id]
        
        # 2. Generate user embedding (simplified - normally from interaction history)
        # Create dummy features for demo with CORRECT dimensions
        text_emb = torch.randn(1, 384).to(device)      # ✅ Sentence-Transformers output
        image_emb = torch.randn(1, 2048).to(device)    # ✅ ResNet-50 output
        behavior_feat = torch.randn(1, 32).to(device)  # ✅ Behavior features
        
        # 3. Get recommendations from model
        with torch.no_grad():
            logits, fusion_weights = model(
                text_emb, image_emb, behavior_feat,
                return_fusion_weights=True
            )
            
            # Get top-K items
            scores = torch.softmax(logits, dim=1)[0]
            top_scores, top_items = torch.topk(scores, request.top_k)
        
        # 4. Get item details
        recommendations = []
        for rank, (item_idx, score) in enumerate(zip(top_items.cpu().numpy(), top_scores.cpu().numpy()), 1):
            item = items_df.iloc[item_idx]
            
            rec_item = RecommendationItem(
                item_id=int(item['item_id']),
                name=item['name'],
                category=item['category'],
                score=float(score),
                rank=rank,
                
                # Explainability
                text_contribution=float(fusion_weights[0, 0]),
                image_contribution=float(fusion_weights[0, 1]),
                behavior_contribution=float(fusion_weights[0, 2]),
                
                # Metadata
                avg_rating=float(item['avg_rating']),
                num_ratings=int(item['num_ratings']),
                price=float(item['price']),
                brand=item['brand']
            )
            
            recommendations.append(rec_item)
        
        # 5. Build response
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response = RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            user_preference_type=user['preference_type'],
            fusion_weights={
                'text': float(fusion_weights[0, 0]),
                'image': float(fusion_weights[0, 1]),
                'behavior': float(fusion_weights[0, 2])
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
        
        # Generate fusion weights (normally loaded from trained model)
        text_emb = torch.randn(1, 384).to(device)      # ✅ Sentence-Transformers
        image_emb = torch.randn(1, 2048).to(device)    # ✅ ResNet-50
        behavior_feat = torch.randn(1, 32).to(device)  # ✅ Behavior features
        
        with torch.no_grad():
            _, fusion_weights = model(
                text_emb, image_emb, behavior_feat,
                return_fusion_weights=True
            )
        
        return UserProfileResponse(
            user_id=user_id,
            age=int(user['age']),
            preference_type=user['preference_type'],
            preferred_categories=user['preferred_categories'],
            registration_date=user['registration_date'],
            fusion_weights={
                'text': float(fusion_weights[0, 0]),
                'image': float(fusion_weights[0, 1]),
                'behavior': float(fusion_weights[0, 2])
            },
            num_interactions=0,  # TODO: Count from interactions
            avg_rating_given=0.0  # TODO: Calculate from interactions
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