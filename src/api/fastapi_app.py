"""
FastAPI Recommendation API
Provides endpoints for personalized recommendations with explainability
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import json
import re
import socket
from collections import deque

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import yaml

from src.models.multimodal_encoder import MultiModalEncoder
from src.models.recommendation_model import FedPerRecommender
from src.vector_db.milvus_manager import MilvusManager
from src.api.metrics_calculator import MetricsCalculator, explain_recommendation
from src.api.inference_service import CatalogEmbeddingCache, UserBehaviorTracker, RecommendationEngine
from src.api.demo_session import DemoSessionManager, OnlinePersonalizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Federated Multi-Modal Recommendation API",
    description="API for personalized recommendations with privacy-preserving federated learning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
milvus_manager = None
items_df = None
users_df = None
device = None
metrics_calculator = None
catalog_cache = None
behavior_tracker = None
recommendation_engine = None
demo_sessions: Optional[DemoSessionManager] = None
online_personalizer: Optional[OnlinePersonalizer] = None
app_config: Dict[str, Any] = {}

PRIVACY_DEMO_SESSION_RE = re.compile(r"^[a-zA-Z0-9_-]{8,64}$")
PRIVACY_DEMO_BUFFER_MAX = 24
privacy_demo_sessions: Dict[str, Dict[str, Any]] = {}


def _privacy_demo_get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in privacy_demo_sessions:
        privacy_demo_sessions[session_id] = {
            "laptops": [],
            "phone": None,
            "buffer": deque(maxlen=PRIVACY_DEMO_BUFFER_MAX),
        }
    return privacy_demo_sessions[session_id]


def _privacy_demo_cleanup_session(session_id: str) -> None:
    s = privacy_demo_sessions.get(session_id)
    if not s:
        return
    laptops = s.get("laptops") or []
    if len(laptops) == 0 and s.get("phone") is None:
        privacy_demo_sessions.pop(session_id, None)


async def _privacy_broadcast_laptops(sess: Dict[str, Any], payload: Dict[str, Any]) -> int:
    """Gửi JSON tới mọi tab laptop đang mở; gỡ socket hỏng."""
    laptops: List[WebSocket] = list(sess.get("laptops") or [])
    ok: List[WebSocket] = []
    for ws in laptops:
        try:
            await ws.send_json(payload)
            ok.append(ws)
        except Exception:
            continue
    sess["laptops"] = ok
    return len(ok)


def _cell_str(row: pd.Series, *keys: str, default: str = "") -> str:
    for k in keys:
        if k not in row.index:
            continue
        v = row[k]
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s and s.lower() not in ("nan", "none", "<na>"):
            return s
    return default


_catalog_title_by_item_id: Optional[Dict[str, str]] = None


def _build_catalog_title_lookup() -> Dict[str, str]:
    """ASIN → tiêu đề dài nhất tìm được trong toàn bộ catalog (nhiều dòng trùng ASIN)."""
    global _catalog_title_by_item_id
    if _catalog_title_by_item_id is not None:
        return _catalog_title_by_item_id
    out: Dict[str, str] = {}
    if items_df is None or len(items_df) == 0 or "item_id" not in items_df.columns:
        return out
    for i in range(len(items_df)):
        r = items_df.iloc[i]
        iid = _cell_str(r, "item_id")
        if not iid:
            continue
        t = _cell_str(r, "item_title", "title", "name", "product_title")
        if t and len(t) > len(out.get(iid, "")):
            out[iid] = t
    _catalog_title_by_item_id = out
    return out


def _preferred_catalog_positions(df: pd.DataFrame) -> np.ndarray:
    """Ưu tiên các dòng có tiêu đề đọ được để demo nhìn giống ‘sản phẩm’ hơn."""
    n = len(df)
    if n == 0:
        return np.array([], dtype=np.int64)
    if "item_title" in df.columns:
        s = df["item_title"].fillna("").astype(str)
        good = s.str.len() > 2
        good &= ~s.str.lower().str.strip().isin(["nan", "none", ""])
        pos = np.flatnonzero(good.to_numpy())
        if pos.size > 0:
            return pos.astype(np.int64)
    return np.arange(n, dtype=np.int64)


def _row_display_fields(row: pd.Series) -> Dict[str, Any]:
    name = _cell_str(row, "item_title", "name", "title", "item_name", "product_title")
    item_id = _cell_str(row, "item_id") or str(row.name)

    if not name or name.startswith("Sản phẩm"):
        lut = _build_catalog_title_lookup()
        if item_id and item_id in lut:
            name = lut[item_id]

    if not name:
        name = f"ASIN {item_id}" if item_id else "Sản phẩm"

    category = (
        _cell_str(row, "item_category", "category", "main_category", "domain", "main_cat") or ""
    )
    if not category:
        raw = _cell_str(row, "categories")
        if raw:
            category = raw.replace("[", "").replace("]", "").split(",")[0].strip()[:48]
    if not category:
        category = "—"

    brand = _cell_str(row, "item_brand", "brand", "manufacturer")
    image_url = _cell_str(row, "item_image_url", "image_url", "image", "thumbnail")

    price = 0.0
    for k in ("item_price", "price"):
        if k in row.index and pd.notna(row[k]):
            try:
                price = float(row[k])
                break
            except (TypeError, ValueError):
                pass

    return {
        "name": name,
        "category": category,
        "brand": brand,
        "price": price,
        "image_url": image_url,
        "item_id": item_id,
    }


class RecommendationRequest(BaseModel):
    """Request for recommendations"""
    user_id: int
    top_k: int = 10
    filters: Optional[Dict[str, str]] = None
    explain: bool = True


class RecommendationItem(BaseModel):
    """Single recommendation item"""
    item_id: str = ""
    name: str = ""
    category: str = ""
    score: float = 0.0
    rank: int = 0

    text_contribution: Optional[float] = None
    image_contribution: Optional[float] = None
    behavior_contribution: Optional[float] = None
    explanation: Optional[str] = None
    
    # Metadata
    avg_rating: float = 0.0
    num_ratings: int = 0
    price: float = 0.0
    brand: str = ""
    image_url: Optional[str] = None
    probability_percent: float = 0.0


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
    output_classes: int = 0
    score_explanation: str = ""


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


class ModelInfoResponse(BaseModel):
    """Expose embedded model names for demos."""
    ok: bool
    device: str
    architecture: str
    text_model: Optional[str] = None
    image_model: Optional[str] = None
    notes: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize models and data on startup"""
    global model, milvus_manager, items_df, users_df, device, metrics_calculator
    global catalog_cache, behavior_tracker, recommendation_engine
    global demo_sessions, online_personalizer
    global app_config
    
    logger.info("🚀 Starting up API server...")

    metrics_calculator = MetricsCalculator()

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
        app_config = config or {}

        logger.info("📂 Loading data...")
        amazon_dir = project_root / "data" / "amazon_2023_processed"
        amazon_client0 = amazon_dir / "client_0" / "data.pkl"

        if amazon_client0.exists():
            amazon_interactions_df = pd.read_pickle(amazon_client0)

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

            items_df = tmp.drop_duplicates(subset=["item_id"]).reset_index(drop=True)
            logger.info(f"✅ Built Amazon item catalog from client_0: {len(items_df)} items")

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

        elif (amazon_dir / "items_global.csv").exists():
            items_df = pd.read_csv(amazon_dir / "items_global.csv")
            logger.warning("⚠️  Loaded Amazon items_global.csv (may not include embeddings for ranking)")
            logger.info(f"✅ Loaded Amazon items table: {len(items_df)} items")
        else:
            items_df = pd.read_csv(project_root / "data/simulated_clients/items_global.csv")
            logger.warning("⚠️  Using simulated items table (Amazon items_global.csv not found)")
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
        
        logger.info("🔌 Connecting to Milvus...")
        try:
            milvus_manager = MilvusManager(
                host="localhost",
                port="19530",
                collection_name="item_embeddings",
                embedding_dim=384
            )
            
            if milvus_manager.collection:
                logger.info("✅ Milvus connected and collection loaded")
            else:
                logger.warning("⚠️  Milvus connected but collection not found. Run milvus_manager.py to create it.")
        except Exception as e:
            logger.warning(f"⚠️  Milvus connection issue: {e}")
            logger.info("💡 You can still use the API, but similarity search won't work")
            milvus_manager = None
        
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
            num_classes=model_cfg.get("num_classes", 5),
        )
        model.to(device)
        model.eval()
        
        ckpt_candidates = sorted(
            project_root.glob("experiments/**/models/global_model_final.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if ckpt_candidates:
            ckpt_path = ckpt_candidates[0]
            try:
                checkpoint = torch.load(str(ckpt_path), map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                logger.info(f"✅ Loaded checkpoint: {ckpt_path}")
            except Exception as e:
                logger.warning(f"⚠️  Could not load checkpoint {ckpt_path}: {e}")
                logger.info("   Using random-initialized model for demo")
        else:
            logger.warning("⚠️  No checkpoint found — using random-initialized model for demo")
        
        logger.info("✅ Model loaded")

        # Initialize inference service for e-commerce
        logger.info("📦 Initializing catalog cache and recommendation engine...")
        try:
            catalog_cache = CatalogEmbeddingCache()
            catalog_cache.load_from_client_data(
                str(project_root / "data" / "processed" / "multi_category"),
                max_clients=40
            )
            behavior_tracker = UserBehaviorTracker()
            recommendation_engine = RecommendationEngine(model, catalog_cache, behavior_tracker)
            demo_sessions = DemoSessionManager(project_root)
            online_personalizer = OnlinePersonalizer(model, device)
            logger.info(f"✅ Recommendation engine ready with {len(catalog_cache.item_embeddings)} cached items!")
        except Exception as e:
            logger.error(f"❌ Failed to initialize recommendation engine: {e}")
            catalog_cache = None
            behavior_tracker = None
            recommendation_engine = None
            demo_sessions = DemoSessionManager(project_root)
            online_personalizer = None

        global _catalog_title_by_item_id
        _catalog_title_by_item_id = None
        
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


@app.get("/api/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Return model names used in this deployment (for demos/UI)."""
    processing = (app_config.get("processing") or {}) if isinstance(app_config, dict) else {}
    text_model = processing.get("text_model")
    image_model = processing.get("image_model")
    arch = type(model).__name__ if model is not None else "unloaded"
    dev = str(device) if device is not None else "unknown"
    notes = "FedPerRecommender + MultiModalEncoder; personal head stays per client/session."
    return ModelInfoResponse(
        ok=True,
        device=dev,
        architecture=arch,
        text_model=str(text_model) if text_model else None,
        image_model=str(image_model) if image_model else None,
        notes=notes,
    )


def _guess_lan_ipv4() -> Optional[str]:
    """IPv4 của máy chủ trên LAN (ước lượng qua socket), phục vụ link mở từ điện thoại."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(0.35)
            s.connect(("8.8.8.8", 80))
            return str(s.getsockname()[0])
    except OSError:
        return None


@app.get("/api/demo-lan-hint")
async def demo_lan_hint(request: Request):
    """
    Gợi ý base URL để điện thoại (cùng Wi‑Fi) mở demo — tránh dùng localhost trên điện thoại.
    """
    port = request.url.port
    if port is None:
        port = 443 if request.url.scheme == "https" else 80
    lan = _guess_lan_ipv4()
    if not lan:
        return {
            "ok": False,
            "lan_ipv4": None,
            "base_for_phone": None,
            "mobile_demo_path": "/demo/mobile_privacy_demo.html",
            "tips": [
                "Không tự nhận diện được IP LAN — vào Cài đặt mạng trên Windows, xem IPv4 của Wi‑Fi, thay vào http://<IP>:8000/demo/...",
                "Trên điện thoại không bao giờ dùng localhost (đó là chính điện thoại).",
            ],
        }
    base = f"{request.url.scheme}://{lan}:{port}"
    return {
        "ok": True,
        "lan_ipv4": lan,
        "port": port,
        "base_for_phone": base,
        "mobile_demo_path": "/demo/mobile_privacy_demo.html",
        "tips": [
            "Điện thoại: dùng IP Wi‑Fi laptop trong URL, không dùng localhost.",
            "Không vào được: kiểm tra firewall cổng 8000.",
        ],
    }


@app.websocket("/ws/privacy-demo/{session_id}")
async def privacy_demo_websocket(websocket: WebSocket, session_id: str, role: str = Query(...)):
    """
    Relay minh họa: điện thoại (role=phone) gửi JSON type=privacy_exchange
    → laptop (role=laptop) nhận cùng payload để hiển thị song song.
    """
    if role not in ("laptop", "phone"):
        await websocket.close(code=4400)
        return
    if not PRIVACY_DEMO_SESSION_RE.match(session_id):
        await websocket.close(code=4401)
        return

    await websocket.accept()
    sess = _privacy_demo_get_session(session_id)

    if role == "laptop":
        laptops: List[WebSocket] = sess.setdefault("laptops", [])
        laptops.append(websocket)
        peer_online = sess.get("phone") is not None
        try:
            await websocket.send_json(
                {
                    "type": "system",
                    "message": "Đã kết nối (màn hình laptop).",
                    "peer_online": peer_online,
                }
            )
            if peer_online and sess.get("phone") is not None:
                try:
                    await sess["phone"].send_json(
                        {
                            "type": "system",
                            "message": "Màn hình laptop đã online.",
                            "peer_online": True,
                        }
                    )
                except Exception:
                    pass
        except Exception:
            pass

        buf = sess["buffer"]
        while buf:
            try:
                await websocket.send_json(buf.popleft())
            except Exception:
                break

        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "ping":
                    try:
                        await websocket.send_json({"type": "pong"})
                    except Exception:
                        break
        except WebSocketDisconnect:
            pass
        finally:
            lst = sess.get("laptops") or []
            if websocket in lst:
                lst.remove(websocket)
            sess["laptops"] = lst
            phone = sess.get("phone")
            if phone is not None:
                try:
                    await phone.send_json(
                        {
                            "type": "system",
                            "message": "Màn hình laptop đã ngắt (có thể còn tab khác).",
                            "peer_online": len(sess.get("laptops") or []) > 0,
                        }
                    )
                except Exception:
                    pass
            _privacy_demo_cleanup_session(session_id)
        return

    # --- phone ---
    old = sess.get("phone")
    if old is not None:
        try:
            await old.close(code=4001, reason="phone_reconnected")
        except Exception:
            pass
    sess["phone"] = websocket

    try:
        n_laptops = len(sess.get("laptops") or [])
        await websocket.send_json(
            {
                "type": "system",
                "message": "Đã kết nối (điện thoại).",
                "peer_online": n_laptops > 0,
            }
        )
        if n_laptops > 0:
            await _privacy_broadcast_laptops(
                sess,
                {
                    "type": "system",
                    "message": "Điện thoại đã online.",
                    "peer_online": True,
                },
            )
    except Exception:
        pass

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "detail": "Invalid JSON"})
                continue

            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if msg.get("type") != "privacy_exchange":
                await websocket.send_json(
                    {"type": "error", "detail": "Chỉ hỗ trợ type=privacy_exchange hoặc ping"}
                )
                continue

            n_sent = await _privacy_broadcast_laptops(sess, msg)
            if n_sent == 0:
                sess["buffer"].append(msg)

            try:
                await websocket.send_json(
                    {
                        "type": "system",
                        "message": "Đã gửi tới màn hình laptop."
                        if n_sent > 0
                        else "Chưa có tab laptop nào mở trang nhận — tin đã xếp hàng.",
                        "peer_online": n_sent > 0,
                    }
                )
            except Exception:
                pass

    except WebSocketDisconnect:
        sess["phone"] = None
        await _privacy_broadcast_laptops(
            sess,
            {
                "type": "system",
                "message": "Điện thoại đã ngắt kết nối.",
                "peer_online": False,
            },
        )
        _privacy_demo_cleanup_session(session_id)


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
        
        user = users_df.iloc[request.user_id]

        if recommendation_engine is not None and catalog_cache is not None and catalog_cache.is_loaded:
            session_id = f"user_session_{request.user_id}"
            res = recommendation_engine.get_recommendations(
                session_id=session_id,
                top_k=request.top_k
            )
            
            recs = []
            for rank, item in enumerate(res["recommendations"], 1):
                recs.append(RecommendationItem(
                    item_id=item["item_id"],
                    name=item["title"],
                    category=item["category"],
                    score=float(item["predicted_rating"]),
                    rank=rank,
                    probability_percent=float(item["confidence"]) * 100.0,
                    text_contribution=float(res["fusion_weights"]["text"]),
                    image_contribution=float(res["fusion_weights"]["image"]),
                    behavior_contribution=float(res["fusion_weights"]["behavior"]),
                    avg_rating=float(item["rating"]),
                    num_ratings=int(item["review_count"]),
                    price=float(item["price"]),
                    image_url=item["image_url"] or None
                ))
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return RecommendationResponse(
                user_id=request.user_id,
                recommendations=recs,
                user_preference_type=str(user.get("preference_type", "amazon_user") if hasattr(user, 'get') else "amazon_user"),
                fusion_weights={
                    "text": float(res["fusion_weights"]["text"]),
                    "image": float(res["fusion_weights"]["image"]),
                    "behavior": float(res["fusion_weights"]["behavior"]),
                },
                timestamp=datetime.now().isoformat(),
                processing_time_ms=processing_time,
                output_classes=res["total_candidates"],
                score_explanation="Sử dụng mô hình FedPer thực tế kết hợp Multimodal Embeddings (Text + Image) và Behavior."
            )
        
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
            n_cls = int(scores.shape[0])
            effective_k = max(1, min(request.top_k, n_cls))
            top_scores, top_items = torch.topk(scores, effective_k)
        
        # 4. Map logits class → catalog rows (ưu tiên dòng có tiêu đề; softmax → hiển thị %)
        recommendations = []
        positions = _preferred_catalog_positions(items_df)
        stride_p = max(1, len(positions) // max(n_cls, 1))
        score_expl = (
            f"Điểm là xác suất softmax trên {n_cls} lớp đầu ra (tổng các lớp = 100%). "
            f"Nếu các lớp gần đều, mỗi lớp ~{100.0 / max(n_cls, 1):.1f}% — "
            "đó là bình thường, không phải ‘điểm chất lượng’ thang 0–100."
        )
        for rank, (item_idx, score) in enumerate(zip(top_items.cpu().numpy(), top_scores.cpu().numpy()), 1):
            cls_i = int(item_idx)
            j = int((cls_i * stride_p) % len(positions))
            idx = int(positions[j])
            row = items_df.iloc[idx]
            disp = _row_display_fields(row)
            prob_pct = round(float(score) * 100.0, 2)

            item_avg_rating = 0.0
            for k in ("avg_rating", "rating"):
                if k in row.index and pd.notna(row[k]):
                    try:
                        item_avg_rating = float(row[k])
                        break
                    except (TypeError, ValueError):
                        pass
            item_num_ratings = 0
            for k in ("num_ratings", "num_interactions"):
                if k in row.index and pd.notna(row[k]):
                    try:
                        item_num_ratings = int(row[k])
                        break
                    except (TypeError, ValueError):
                        pass

            rec_item = RecommendationItem(
                item_id=disp["item_id"],
                name=disp["name"],
                category=disp["category"],
                score=float(score),
                rank=rank,
                probability_percent=prob_pct,
                
                # Explainability
                text_contribution=float(fusion_weights[0, 0]),
                image_contribution=float(fusion_weights[0, 1]),
                behavior_contribution=float(fusion_weights[0, 2]),
                
                # Metadata
                avg_rating=item_avg_rating,
                num_ratings=item_num_ratings,
                price=disp["price"],
                brand=disp["brand"],
                image_url=disp["image_url"] or None,
            )
            
            recommendations.append(rec_item)
        
        # 5. Build response
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response = RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            user_preference_type=str(user.get("preference_type", "amazon_user") if hasattr(user, 'get') else "amazon_user"),
            fusion_weights={
                "text": float(fusion_weights[0, 0].item()),
                "image": float(fusion_weights[0, 1].item()),
                "behavior": float(fusion_weights[0, 2].item()),
            },
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time,
            output_classes=n_cls,
            score_explanation=score_expl,
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
        # Resolve column names (Amazon vs synthetic)
        cat_col = 'item_category' if 'item_category' in items_df.columns else 'category'
        categories = items_df[cat_col].value_counts().to_dict() if cat_col in items_df.columns else {}
        
        # Users preference types (may not exist in Amazon data)
        pref_types = {}
        if 'preference_type' in users_df.columns:
            pref_types = users_df['preference_type'].value_counts().to_dict()
        
        # Average rating (may be avg_rating or computed from data)
        avg_rating = 0.0
        if 'avg_rating' in items_df.columns:
            avg_rating = float(items_df['avg_rating'].mean())
        elif 'rating' in items_df.columns:
            avg_rating = float(items_df['rating'].mean())
        
        total_ratings = 0
        if 'num_ratings' in items_df.columns:
            total_ratings = int(items_df['num_ratings'].sum())
        elif 'num_interactions' in users_df.columns:
            total_ratings = int(users_df['num_interactions'].sum())
        else:
            total_ratings = len(items_df)
        
        stats = {
            "total_items": len(items_df),
            "total_users": len(users_df),
            "categories": categories,
            "preference_types": pref_types,
            "avg_item_rating": avg_rating,
            "total_ratings": total_ratings,
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


@app.get("/metrics")
async def get_performance_metrics():
    """
    Get comprehensive performance metrics with formulas and explanations
    
    Returns detailed metrics for:
    - Privacy score (with breakdown)
    - Latency metrics (with comparison)
    - Personalization score (with explanation)
    - Recommendation quality (Precision, Recall, NDCG)
    """
    try:
        # Generate sample recommendations to calculate metrics
        userId = 0
        user = users_df.iloc[userId]
        
        # Get recommendations (deterministic for user 0)
        torch.manual_seed(userId)
        text_emb = torch.randn(1, 384).to(device)
        torch.manual_seed(userId + 10000)
        image_emb = torch.randn(1, 2048).to(device)
        torch.manual_seed(userId + 20000)
        behavior_feat = torch.randn(1, 32).to(device)
        
        with torch.no_grad():
            logits, fusion_weights = model(
                text_emb, image_emb, behavior_feat,
                return_fusion_weights=True
            )
            
            scores = torch.softmax(logits, dim=1)[0]
            top_k = min(10, scores.shape[0])
            top_scores, top_items = torch.topk(scores, top_k)
        
        # Prepare recommendations for metrics (resolve column names)
        recommendations = []
        for rank, (item_idx, score) in enumerate(zip(top_items.cpu().numpy(), top_scores.cpu().numpy()), 1):
            idx = int(item_idx) % len(items_df)
            item = items_df.iloc[idx]
            recommendations.append({
                'item_id': str(item.get('item_id', idx)),
                'name': str(item.get('item_title', item.get('name', f'Item {idx}')) or f'Item {idx}'),
                'category': str(item.get('item_category', item.get('category', 'Unknown')) or 'Unknown'),
                'score': float(score),
                'rank': rank,
                'avg_rating': float(item.get('avg_rating', item.get('rating', 3.5)) or 3.5),
                'num_ratings': int(item.get('num_ratings', 0) or 0),
                'price': float(item.get('item_price', item.get('price', 0.0)) or 0.0),
                'brand': str(item.get('item_brand', item.get('brand', '')) or '')
            })
        
        # User preference type (may not exist in Amazon data)
        user_pref = str(user.get('preference_type', 'amazon_user') if hasattr(user, 'get') else 'amazon_user')
        
        # Generate comprehensive metrics report
        report = metrics_calculator.generate_comprehensive_report(
            recommendations=recommendations,
            fusion_weights={
                'text': float(fusion_weights[0, 0]),
                'image': float(fusion_weights[0, 1]),
                'behavior': float(fusion_weights[0, 2])
            },
            user_preference_type=user_pref,
            processing_time_ms=25.5,  # Typical inference time
            num_local_updates=5
        )
        
        return report
        
    except Exception as e:
        logger.error(f"❌ Get metrics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ======================================================================
# E-commerce & Demo Auth Endpoints
# ======================================================================

def _resolve_demo_session(session_id: str) -> Optional[Dict[str, Any]]:
    if demo_sessions is None:
        return None
    return demo_sessions.get_session(session_id)


def _format_catalog_products(meta_list: list) -> list:
    out = []
    for m in meta_list:
        out.append({
            "id": m.get("id", m.get("item_id", "")),
            "title": m.get("title", "Sản phẩm"),
            "description": m.get("description", ""),
            "category": m.get("category", ""),
            "price": m.get("price", 0),
            "rating": m.get("rating", 4),
            "review_count": m.get("review_count", 0),
            "image_url": m.get("image_url", ""),
            "image_fallback": m.get("image_fallback", ""),
        })
    return out


@app.get("/api/auth/clients")
async def list_federated_clients():
    """40 federated clients — mỗi client = một silo / chi nhánh."""
    if demo_sessions is None:
        raise HTTPException(status_code=503, detail="Demo session manager not ready")
    return {"clients": demo_sessions.list_clients()}


@app.get("/api/auth/clients/{client_id}/users")
async def list_client_users(client_id: int, limit: int = 12):
    if demo_sessions is None:
        raise HTTPException(status_code=503, detail="Demo session manager not ready")
    if client_id < 0 or client_id > 39:
        raise HTTPException(status_code=400, detail="client_id must be 0–39")
    return {"users": demo_sessions.list_users_for_client(client_id, limit=limit)}


@app.post("/api/auth/login")
async def demo_login(payload: dict):
    """
    Đăng nhập với federated client + khách hàng (user trong silo dữ liệu).
    Trả session_id dùng cho recommend / track / cập nhật personal head.
    """
    if demo_sessions is None:
        raise HTTPException(status_code=503, detail="Demo session manager not ready")

    client_id = int(payload.get("client_id", 0))
    user_id = payload.get("user_id")
    try:
        sess = demo_sessions.create_session(client_id, user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    products = []
    if catalog_cache and catalog_cache.is_loaded:
        products = catalog_cache.get_products_for_client(
            client_id, category=sess.get("category"), limit=80
        )

    if online_personalizer is not None:
        online_personalizer.ensure_client(client_id)

    return {
        **sess,
        "products": products,
    }


@app.get("/api/session/{session_id}")
async def get_demo_session_info(session_id: str):
    sess = _resolve_demo_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    fl_stats = {}
    if online_personalizer is not None:
        fl_stats = online_personalizer.get_stats(session_id)
    return {**sess, "fl_stats": fl_stats}


@app.post("/api/fl-update")
async def trigger_fl_update(payload: dict):
    """Chạy vài bước cập nhật personal head từ hành vi session (FedPer local)."""
    if online_personalizer is None:
        raise HTTPException(status_code=503, detail="Online personalizer not ready")

    session_id = payload.get("session_id", "default")
    result = online_personalizer.run_local_update(session_id)
    return result


@app.get("/api/products")
async def get_products(
    category: str = None,
    page: int = 1,
    limit: int = 20,
    session_id: str = None,
    client_id: int = None,
):
    """Get product catalog with optional category / federated client filter."""
    if catalog_cache is None or not catalog_cache.is_loaded:
        raise HTTPException(status_code=503, detail="Catalog embedding cache not initialized or not loaded")

    sess = _resolve_demo_session(session_id) if session_id else None
    cid = client_id
    cat = category
    if sess:
        cid = sess.get("client_id", cid)
        cat = cat or sess.get("category")

    if cid is not None:
        items = catalog_cache.get_products_for_client(cid, category=cat, limit=500)
    else:
        items = _format_catalog_products(list(catalog_cache.items_metadata.values()))
        if cat:
            items = [i for i in items if cat.lower() in i.get("category", "").lower()]

    start = (page - 1) * limit
    page_items = items[start : start + limit]
    return {
        "products": page_items,
        "total": len(items),
        "page": page,
        "total_pages": max(1, (len(items) + limit - 1) // limit),
        "client_id": cid,
    }

@app.get("/api/products/{product_id}")
async def get_product(product_id: str):
    """Get single product details."""
    if catalog_cache is None or not catalog_cache.is_loaded:
        raise HTTPException(status_code=503, detail="Catalog embedding cache not initialized or not loaded")
        
    meta = catalog_cache.items_metadata.get(product_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Product not found")
    return meta

@app.post("/api/track-behavior")
async def track_behavior(data: dict):
    """Track user behavior events; queue sample for on-device personal-head update."""
    if behavior_tracker is None:
        raise HTTPException(status_code=503, detail="Behavior tracker not initialized")

    session_id = data.get("session_id", "default")
    event_type = data.get("event_type") or data.get("action", "view")
    item_id = data.get("item_id") or data.get("product_id")

    behavior_tracker.record_event(session_id, event_type, item_id, data)

    fl_result = {"updated": False}
    sess = _resolve_demo_session(session_id)
    client_id = sess.get("client_id") if sess else data.get("client_id")

    if (
        online_personalizer is not None
        and catalog_cache is not None
        and item_id
        and item_id in catalog_cache.item_embeddings
    ):
        emb = catalog_cache.item_embeddings[item_id]
        behavior_vec = behavior_tracker.get_behavior_vector(session_id)
        cid = int(client_id) if client_id is not None else 0
        online_personalizer.queue_interaction(
            session_id,
            cid,
            str(item_id),
            event_type,
            text_emb=emb["text_emb"],
            image_emb=emb["image_emb"],
            behavior_vec=behavior_vec,
        )
        fl_result = online_personalizer.run_local_update(session_id, min_samples=2, steps=2)

    return {"status": "ok", "fl_update": fl_result}

@app.post("/api/track-behavior/batch")
async def track_behavior_batch(data: dict):
    """Track batch user behavior events."""
    if behavior_tracker is None:
        raise HTTPException(status_code=503, detail="Behavior tracker not initialized")
        
    session_id = data.get('session_id', 'default')
    behavior = data.get('behavior', {})
    
    # Process clicks
    clicks = behavior.get('clicks', {})
    for item_id, count in clicks.items():
        for _ in range(int(count)):
            behavior_tracker.record_event(session_id, 'click', item_id)
            
    # Process views/durations
    views = behavior.get('views', {})
    for item_id, duration in views.items():
        behavior_tracker.record_event(session_id, 'view', item_id, {"duration": duration})
        
    # Process cart additions
    cart_adds = behavior.get('cart_adds', {})
    for item_id, count in cart_adds.items():
        for _ in range(int(count)):
            behavior_tracker.record_event(session_id, 'cart', item_id)
            
    # Process search queries
    search_queries = behavior.get('search_queries', [])
    for q_item in search_queries:
        query = q_item.get('query', '')
        if query:
            behavior_tracker.record_event(session_id, 'search', data={"query": query})
            
    return {"status": "ok"}

@app.post("/api/recommend")
async def get_recommendations_v2(data: dict):
    """Get AI-powered recommendations using real model inference."""
    if recommendation_engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not initialized")

    session_id = data.get("session_id", "default")
    context_item = (
        data.get("context_item_id")
        or data.get("product_context")
        or data.get("context_item")
    )
    category = data.get("category")
    top_k = data.get("top_k", 10)

    sess = _resolve_demo_session(session_id)
    client_id = sess.get("client_id") if sess else data.get("client_id")
    if sess and not category:
        category = sess.get("category")

    inference_model = None
    if online_personalizer is not None and client_id is not None:
        inference_model = online_personalizer.get_model_for_inference(int(client_id))

    result = recommendation_engine.get_recommendations(
        session_id=session_id,
        context_item_id=context_item,
        category=category,
        client_id=int(client_id) if client_id is not None else None,
        top_k=top_k,
        inference_model=inference_model,
    )

    recs = result.get("recommendations", [])
    products = [
        {
            "id": r.get("item_id"),
            "title": r.get("title"),
            "category": r.get("category"),
            "price": r.get("price"),
            "rating": r.get("rating"),
            "review_count": r.get("review_count", 0),
            "image_url": r.get("image_url", ""),
            "image_fallback": (
                catalog_cache.items_metadata.get(r.get("item_id"), {}).get("image_fallback", "")
                if catalog_cache
                else ""
            ),
            "description": catalog_cache.items_metadata.get(r.get("item_id"), {}).get(
                "description", ""
            )
            if catalog_cache
            else "",
        }
        for r in recs
    ]
    result["recommendations"] = products
    if online_personalizer is not None:
        result["fl_stats"] = online_personalizer.get_stats(session_id)
    return result

@app.get("/api/categories")
async def get_categories():
    """Get available product categories."""
    if catalog_cache is None or not catalog_cache.is_loaded:
        raise HTTPException(status_code=503, detail="Catalog embedding cache not initialized or not loaded")
        
    categories = {}
    for item in catalog_cache.items_metadata.values():
        cat = item.get('category', 'Unknown')
        categories[cat] = categories.get(cat, 0) + 1
    return {"categories": [{"name": k, "count": v} for k, v in categories.items()]}

@app.get("/api/search")
async def search_products(q: str, limit: int = 20):
    """Search products by query."""
    if catalog_cache is None or not catalog_cache.is_loaded:
        raise HTTPException(status_code=503, detail="Catalog embedding cache not initialized or not loaded")
        
    q_lower = q.lower()
    results = [
        item for item in catalog_cache.items_metadata.values()
        if q_lower in item.get('title', '').lower() or q_lower in item.get('category', '').lower()
    ]
    formatted = _format_catalog_products(results[:limit])
    return {"products": formatted, "results": formatted, "total": len(results)}


_demo_dir = Path(__file__).resolve().parent.parent.parent / "demo"
if _demo_dir.is_dir():
    app.mount("/demo", StaticFiles(directory=str(_demo_dir), html=True), name="demo_static")

# Mount e-commerce frontend
ecommerce_dir = Path(__file__).resolve().parent.parent.parent / "ecommerce"
if ecommerce_dir.exists():
    app.mount("/shop", StaticFiles(directory=str(ecommerce_dir), html=True), name="shop")


def get_uvicorn_server_kwargs(*, reload: bool) -> Dict[str, Any]:
    """
    Cấu hình Uvicorn giảm WebSocket đóng bất thường (mã 1006): keep-alive, ping/pong WS, tắt deflate.
    Dùng backend ``websockets`` nếu đã cài gói ``websockets`` (ổn định hơn wsproto trên một số máy Windows).
    """
    kwargs: Dict[str, Any] = {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": reload,
        "log_level": "info",
        "timeout_keep_alive": 300,
        "ws_ping_interval": 25.0,
        "ws_ping_timeout": 300.0,
        "ws_per_message_deflate": False,
    }
    try:
        import websockets  # noqa: F401

        kwargs["ws"] = "websockets"
    except ImportError:
        pass
    return kwargs


if __name__ == "__main__":
    import os
    import uvicorn

    print("=" * 70)
    print("FEDERATED MULTI-MODAL RECOMMENDATION API")
    print("=" * 70)
    print("Starting server...")
    print("Docs available at: http://localhost:8000/docs")
    _reload = os.environ.get("FED_REC_API_RELOAD", "").strip().lower() in ("1", "true", "yes")
    print("Reload:", "ON" if _reload else "OFF (set FED_REC_API_RELOAD=1 to enable)")
    print("=" * 70)

    uvicorn.run("src.api.fastapi_app:app", **get_uvicorn_server_kwargs(reload=_reload))