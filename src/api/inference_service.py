"""
Inference Service for FedPer Recommendation Model.
Replaces random embeddings with real catalog-based inference.

Components:
- CatalogEmbeddingCache: Pre-load embeddings from processed client data
- UserBehaviorTracker: Track user clicks/views/cart for behavior features
- RecommendationEngine: Score items using the real FedPer model
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import logging
import time
from pathlib import Path
from collections import defaultdict
from typing import Optional

from src.api.demo_catalog import (
    client_category,
    enrich_product_metadata,
)

logger = logging.getLogger(__name__)


class CatalogEmbeddingCache:
    """Pre-computed embeddings for all catalog items."""

    def __init__(self):
        self.item_embeddings = {}   # item_id -> {text_emb, image_emb}
        self.category_items = defaultdict(list)  # category -> [item_ids]
        self.client_items = defaultdict(list)   # client_id -> [item_ids]
        self.items_metadata = {}    # item_id -> {id, title, category, rating, price, ...}
        self._loaded = False

    @property
    def is_loaded(self):
        return self._loaded and len(self.item_embeddings) > 0

    # ------------------------------------------------------------------
    # Category mapping (client index -> Amazon category)
    # ------------------------------------------------------------------
    CATEGORY_MAP = {
        range(0, 10): "All_Beauty",
        range(10, 20): "Video_Games",
        range(20, 30): "Amazon_Fashion",
        range(30, 40): "Baby_Products",
    }

    CATEGORY_NAMES_VI = {
        "All_Beauty": "Làm Đẹp",
        "Video_Games": "Video Games",
        "Amazon_Fashion": "Thời Trang",
        "Baby_Products": "Mẹ & Bé",
    }

    CATEGORY_SAMPLE_TITLES = {
        "All_Beauty": [
            "Son Dưỡng Môi Hồng Tự Nhiên", "Kem Chống Nắng SPF50+", "Serum Vitamin C 20%",
            "Mặt Nạ Dưỡng Ẩm Collagen", "Nước Tẩy Trang Micellar", "Kem Dưỡng Da Ban Đêm",
            "Phấn Phủ Mịn Lì", "Mascara Dày Mi 10x", "Toner Hoa Hồng Organic",
            "Kem Lót Trang Điểm Mịn Da", "Sữa Rửa Mặt Trà Xanh", "Bông Tẩy Trang Cotton",
        ],
        "Video_Games": [
            "PlayStation 5 DualSense Controller", "Nintendo Switch OLED", "Xbox Game Pass Ultimate 12M",
            "Tai Nghe Gaming RGB 7.1", "Bàn Phím Cơ Cherry MX", "Chuột Gaming Wireless 25600 DPI",
            "Ghế Gaming Ergonomic Pro", "Webcam 4K Streaming", "Tay Cầm Bluetooth Mobile",
            "Đĩa Game God of War Ragnarök", "SSD NVMe 1TB Gaming", "Card Đồ Họa RTX 4060",
        ],
        "Amazon_Fashion": [
            "Áo Thun Oversize Basic", "Quần Jeans Slim Fit", "Giày Sneaker Trắng Classic",
            "Túi Xách Da Thật Vintage", "Kính Mát Polarized UV400", "Đồng Hồ Thông Minh",
            "Áo Khoác Bomber Unisex", "Váy Midi Hoa Nhí", "Balo Laptop Chống Nước",
            "Mũ Lưỡi Trai Baseball", "Ví Da Nam Cao Cấp", "Dây Chuyền Bạc Nguyên Chất",
        ],
        "Baby_Products": [
            "Bình Sữa Cho Bé 240ml", "Tã Dán Sơ Sinh Size S", "Xe Đẩy Gấp Gọn Du Lịch",
            "Ghế Ăn Dặm Đa Năng", "Đồ Chơi Xếp Hình Gỗ", "Quần Áo Sơ Sinh Cotton",
            "Máy Hâm Sữa Thông Minh", "Nôi Điện Tự Động", "Bộ Chăm Sóc Trẻ Sơ Sinh",
            "Núm Ti Silicone Mềm", "Khăn Ướt Không Mùi", "Sữa Tắm Gội Cho Bé",
        ],
    }

    def _client_to_category(self, client_idx: int) -> str:
        """Map client index to Amazon category."""
        for rng, cat in self.CATEGORY_MAP.items():
            if client_idx in rng:
                return cat
        return "Unknown"

    # ------------------------------------------------------------------
    def load_from_client_data(self, data_dir: str, max_clients: int = 8):
        """Load embeddings from processed client data pickle files."""
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.warning(f"Data dir not found: {data_dir}")
            return 0

        loaded_items = 0
        title_counters = defaultdict(int)  # category -> index into sample titles

        client_dirs = list(data_path.glob("client_*"))
        client_dirs.sort(key=lambda d: int(d.name.split("_")[1]))

        for client_dir in client_dirs:
            client_idx = int(client_dir.name.split("_")[1])
            if client_idx >= max_clients:
                break

            data_file = client_dir / "data.pkl"
            if not data_file.exists():
                continue

            category = self._client_to_category(client_idx)

            try:
                with open(data_file, "rb") as f:
                    client_data = pickle.load(f)

                items_from_client = self._extract_items(
                    client_data, client_idx, category, title_counters
                )
                loaded_items += items_from_client
                logger.info(
                    f"Loaded {items_from_client} items from {client_dir.name} "
                    f"(category={category})"
                )
            except Exception as e:
                logger.warning(f"Error loading {client_dir}: {e}")
                continue

        self._loaded = True
        logger.info(f"✅ Catalog cache loaded: {loaded_items} items total")
        return loaded_items

    # ------------------------------------------------------------------
    def _extract_items(self, client_data, client_idx, category, title_counters):
        """Extract items from a single client's data.pkl."""
        count = 0

        # --- Format C: Pandas DataFrame ---
        import pandas as pd
        if isinstance(client_data, pd.DataFrame):
            unique_items = client_data.drop_duplicates(subset=["item_id"])
            max_per_client = 150  # Cap per client to save memory

            for _, row in unique_items.head(max_per_client).iterrows():
                item_id = str(row["item_id"])

                try:
                    text_emb = np.asarray(row["text_embedding"], dtype=np.float32)
                    image_emb = np.asarray(row["image_embedding"], dtype=np.float32)

                    if len(text_emb) < 384 or len(image_emb) < 2048:
                        continue

                    self.item_embeddings[item_id] = {
                        "text_emb": torch.tensor(text_emb[:384], dtype=torch.float32),
                        "image_emb": torch.tensor(image_emb[:2048], dtype=torch.float32),
                    }

                    label_val = int(row["label"]) if "label" in row else (int(row["rating"]) - 1 if "rating" in row else 3)
                    raw_title = ""
                    for col in ("item_title", "title", "review_text"):
                        if col in row.index and pd.notna(row[col]):
                            raw_title = str(row[col])[:80]
                            break
                    price = self._generate_price(category, label_val)
                    raw_image = ""
                    for col in ("item_image_url", "image_url", "image"):
                        if col in row.index and pd.notna(row[col]):
                            raw_image = str(row[col]).strip()
                            if raw_image.startswith("http"):
                                break
                            raw_image = ""
                    meta = enrich_product_metadata(
                        item_id,
                        category,
                        fallback_title=raw_title or None,
                        fallback_price=price,
                        fallback_rating=min(label_val + 1, 5),
                        fallback_image_url=raw_image or None,
                    )
                    meta["description"] = meta.get("description", "")
                    self.items_metadata[item_id] = meta
                    self.category_items[category].append(item_id)
                    self.client_items[client_idx].append(item_id)
                    count += 1
                except Exception as e:
                    logger.debug(f"Error extracting item {item_id}: {e}")
                    continue

            return count

        if not isinstance(client_data, dict):
            return 0

        # --- Format A: {train: TensorDataset, test: TensorDataset} ---
        for split_key in ("train", "test"):
            split = client_data.get(split_key)
            if split is None:
                continue

            tensors = None
            if hasattr(split, "tensors"):
                tensors = split.tensors
            elif isinstance(split, (list, tuple)) and len(split) >= 2:
                tensors = split

            if tensors is None or len(tensors) < 2:
                continue

            features_t = tensors[0]
            labels_t = tensors[1]
            max_per_split = 150  # cap per split to save memory

            for i in range(min(len(features_t), max_per_split)):
                feat = features_t[i]
                if isinstance(feat, torch.Tensor):
                    feat = feat.detach().cpu().numpy()
                feat = np.asarray(feat, dtype=np.float32)

                if len(feat) >= 2464:          # 384 + 2048 + 32
                    text_emb = feat[:384]
                    image_emb = feat[384:2432]
                elif len(feat) >= 384:
                    text_emb = feat[:384]
                    image_emb = np.zeros(2048, dtype=np.float32)
                else:
                    continue

                label_val = labels_t[i]
                if isinstance(label_val, torch.Tensor):
                    label_val = label_val.item()
                label_val = int(label_val)

                item_id = f"item_{client_idx}_{split_key}_{i}"

                self.item_embeddings[item_id] = {
                    "text_emb": torch.tensor(text_emb, dtype=torch.float32),
                    "image_emb": torch.tensor(image_emb, dtype=torch.float32),
                }

                # Generate realistic metadata
                price = self._generate_price(category, label_val)
                meta = enrich_product_metadata(
                    item_id,
                    category,
                    fallback_price=price,
                    fallback_rating=min(label_val + 1, 5),
                )
                self.items_metadata[item_id] = meta
                self.category_items[category].append(item_id)
                self.client_items[client_idx].append(item_id)
                count += 1

        # --- Format B: {features: [...], labels: [...]} ---
        if "features" in client_data and count == 0:
            features = client_data["features"]
            labels = client_data.get("labels", [])
            for i, feat in enumerate(features[:200]):
                if isinstance(feat, dict):
                    text_emb = np.asarray(
                        feat.get("text_embedding", np.zeros(384)), dtype=np.float32
                    )
                    image_emb = np.asarray(
                        feat.get("image_embedding", np.zeros(2048)), dtype=np.float32
                    )
                elif isinstance(feat, (np.ndarray, torch.Tensor)):
                    feat_np = feat.numpy() if isinstance(feat, torch.Tensor) else np.asarray(feat)
                    text_emb = feat_np[:384] if len(feat_np) >= 384 else np.zeros(384, dtype=np.float32)
                    image_emb = feat_np[384:2432] if len(feat_np) >= 2432 else np.zeros(2048, dtype=np.float32)
                else:
                    continue

                item_id = f"item_{client_idx}_feat_{i}"
                label_val = int(labels[i]) if i < len(labels) else 3

                self.item_embeddings[item_id] = {
                    "text_emb": torch.tensor(text_emb, dtype=torch.float32),
                    "image_emb": torch.tensor(image_emb, dtype=torch.float32),
                }

                price = self._generate_price(category, label_val)
                meta = enrich_product_metadata(
                    item_id,
                    category,
                    fallback_price=price,
                    fallback_rating=min(label_val + 1, 5),
                )
                self.items_metadata[item_id] = meta
                self.category_items[category].append(item_id)
                self.client_items[client_idx].append(item_id)
                count += 1

        return count

    def get_products_for_client(
        self, client_id: int, category: Optional[str] = None, limit: int = 200
    ) -> list:
        """Items belonging to a federated client's silo (same domain)."""
        ids = self.client_items.get(client_id) or []
        if not ids and category:
            ids = self.category_items.get(category, [])
        out = []
        for iid in ids[:limit]:
            meta = self.items_metadata.get(iid)
            if meta:
                out.append(self._format_product(meta))
        return out

    @staticmethod
    def _format_product(meta: dict) -> dict:
        return {
            "id": meta.get("id", meta.get("item_id", "")),
            "title": meta.get("title", "Sản phẩm"),
            "description": meta.get("description", ""),
            "category": meta.get("category", ""),
            "price": meta.get("price", 0),
            "rating": meta.get("rating", 4),
            "review_count": meta.get("review_count", 0),
            "image_url": meta.get("image_url", ""),
            "image_fallback": meta.get("image_fallback", ""),
        }

    @staticmethod
    def _generate_price(category: str, rating: int) -> int:
        """Generate a realistic price in VND based on category."""
        base_ranges = {
            "All_Beauty": (50_000, 800_000),
            "Video_Games": (200_000, 15_000_000),
            "Amazon_Fashion": (100_000, 3_000_000),
            "Baby_Products": (50_000, 2_000_000),
        }
        lo, hi = base_ranges.get(category, (50_000, 1_000_000))
        # Higher-rated items tend to be slightly more expensive
        factor = 0.8 + rating * 0.1
        price = int(np.random.randint(lo, hi) * factor)
        # Round to nearest 1,000
        return (price // 1000) * 1000


# ======================================================================
class UserBehaviorTracker:
    """Track user behavior within sessions for recommendation."""

    def __init__(self):
        self.sessions = {}

    def _get_session(self, session_id: str) -> dict:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "viewed_items": [],
                "clicked_items": [],
                "carted_items": [],
                "search_queries": [],
                "view_durations": {},
                "created_at": time.time(),
            }
        return self.sessions[session_id]

    @staticmethod
    def _normalize_event_type(event_type: str) -> str:
        if event_type in ("add_to_cart", "cart_add"):
            return "cart"
        return event_type

    def record_event(
        self, session_id: str, event_type: str, item_id: str = None, data: dict = None
    ):
        event_type = self._normalize_event_type(event_type)
        session = self._get_session(session_id)
        if event_type == "view" and item_id:
            if item_id not in session["viewed_items"]:
                session["viewed_items"].append(item_id)
        elif event_type == "click" and item_id:
            session["clicked_items"].append(item_id)
        elif event_type == "cart" and item_id:
            session["carted_items"].append(item_id)
        elif event_type == "search" and data:
            session["search_queries"].append(data.get("query", ""))

    def get_behavior_vector(self, session_id: str) -> torch.Tensor:
        """Generate 32-dim behavior vector from session history."""
        session = self._get_session(session_id)
        behavior = np.zeros(32, dtype=np.float32)

        # Engagement features (dims 0-7)
        behavior[0] = min(len(session["viewed_items"]) / 20.0, 1.0)
        behavior[1] = min(len(session["clicked_items"]) / 10.0, 1.0)
        behavior[2] = min(len(session["carted_items"]) / 5.0, 1.0)
        behavior[3] = min(len(session["search_queries"]) / 5.0, 1.0)

        duration = time.time() - session["created_at"]
        behavior[4] = min(duration / 600.0, 1.0)

        if len(session["viewed_items"]) > 0:
            behavior[5] = len(session["clicked_items"]) / len(session["viewed_items"])
        if len(session["clicked_items"]) > 0:
            behavior[6] = len(session["carted_items"]) / len(session["clicked_items"])
        behavior[7] = min(len(set(session["viewed_items"])) / 10.0, 1.0)

        # Deterministic filler for remaining dims
        np.random.seed(hash(session_id) % (2**31))
        behavior[8:] = np.random.randn(24).astype(np.float32) * 0.1

        return torch.tensor(behavior, dtype=torch.float32)


# ======================================================================
class RecommendationEngine:
    """Score catalog items with the FedPer model and return top-K."""

    def __init__(
        self,
        model,
        catalog_cache: CatalogEmbeddingCache,
        behavior_tracker: UserBehaviorTracker,
    ):
        self.model = model
        self.catalog = catalog_cache
        self.behavior = behavior_tracker
        if model is not None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_recommendations(
        self,
        session_id: str,
        context_item_id: str = None,
        category: str = None,
        client_id: Optional[int] = None,
        top_k: int = 10,
        inference_model: Optional[nn.Module] = None,
    ) -> dict:
        start_time = time.time()
        active_model = inference_model if inference_model is not None else self.model

        if not active_model or not self.catalog.is_loaded:
            return self._fallback_recommendations(category, top_k, client_id=client_id)

        # 1. Behavior vector
        behavior_vec = (
            self.behavior.get_behavior_vector(session_id)
            .unsqueeze(0)
            .to(self.device)
        )

        # 2. Context embedding (average of recently viewed)
        session = self.behavior._get_session(session_id)
        viewed_ids = session["viewed_items"][-5:]

        context_text, context_image = self._build_context(
            viewed_ids, context_item_id
        )

        # 3. Candidate items (prefer federated client silo)
        candidate_ids: list = []
        if client_id is not None:
            candidate_ids = list(self.catalog.client_items.get(client_id, []))
        if not candidate_ids:
            candidate_ids = list(self.catalog.item_embeddings.keys())
        if category:
            cat_items = self.catalog.category_items.get(category, [])
            if cat_items:
                candidate_ids = cat_items if not client_id else [
                    c for c in candidate_ids if c in set(cat_items)
                ] or cat_items

        viewed_set = set(session["viewed_items"])
        candidate_ids = [c for c in candidate_ids if c not in viewed_set]

        # 4. Score
        scores = self._score_items(
            candidate_ids[:500],
            context_text,
            context_image,
            behavior_vec,
            model=active_model,
        )
        scores.sort(key=lambda x: x[1], reverse=True)
        top_items = scores[:top_k]

        # 5. Format
        recommendations = []
        for item_id, score, probs, fw in top_items:
            meta = self.catalog.items_metadata.get(item_id, {})
            recommendations.append(
                {
                    "item_id": item_id,
                    "title": meta.get("title", "Sản phẩm"),
                    "category": meta.get("category", "Unknown"),
                    "category_vi": meta.get("category_vi", ""),
                    "price": meta.get("price", 0),
                    "rating": meta.get("rating", 3),
                    "review_count": meta.get("review_count", 0),
                    "image_url": meta.get("image_url", ""),
                    "predicted_rating": round(score, 2),
                    "confidence": round(float(probs.max()), 3),
                }
            )

        # Fusion weights from last scored item
        fw_dict = {"text": 0.33, "image": 0.33, "behavior": 0.34}
        if top_items:
            last_fw = top_items[0][3]
            if last_fw is not None:
                try:
                    fw = last_fw[0].cpu().numpy()
                    fw_dict = {
                        "text": round(float(fw[0]), 3),
                        "image": round(float(fw[1]), 3),
                        "behavior": round(float(fw[2]), 3),
                    }
                except Exception:
                    pass

        version = "fedper_multi_category"
        if client_id is not None:
            version += f"_client_{client_id}"

        return {
            "recommendations": recommendations,
            "fusion_weights": fw_dict,
            "inference_time_ms": round((time.time() - start_time) * 1000, 1),
            "total_candidates": len(candidate_ids),
            "model_version": version,
            "client_id": client_id,
        }

    # ------------------------------------------------------------------
    def _build_context(self, viewed_ids, context_item_id):
        """Average embedding of recently viewed items."""
        text_embs, image_embs = [], []

        for vid in viewed_ids:
            if vid in self.catalog.item_embeddings:
                text_embs.append(self.catalog.item_embeddings[vid]["text_emb"])
                image_embs.append(self.catalog.item_embeddings[vid]["image_emb"])

        if not text_embs and context_item_id and context_item_id in self.catalog.item_embeddings:
            d = self.catalog.item_embeddings[context_item_id]
            text_embs.append(d["text_emb"])
            image_embs.append(d["image_emb"])

        if text_embs:
            ctx_t = torch.stack(text_embs).mean(0).unsqueeze(0).to(self.device)
            ctx_i = torch.stack(image_embs).mean(0).unsqueeze(0).to(self.device)
        else:
            ctx_t = torch.zeros(1, 384).to(self.device)
            ctx_i = torch.zeros(1, 2048).to(self.device)

        return ctx_t, ctx_i

    def _score_items(
        self, candidate_ids, context_text, context_image, behavior_vec, model=None
    ):
        """Run model forward pass for each candidate and return scores."""
        results = []
        active = model if model is not None else self.model
        rating_values = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32).to(
            self.device
        )

        for item_id in candidate_ids:
            item_data = self.catalog.item_embeddings[item_id]

            # Blend context with item embedding (70% item, 30% context)
            text_in = (
                context_text * 0.3
                + item_data["text_emb"].unsqueeze(0).to(self.device) * 0.7
            )
            image_in = (
                context_image * 0.3
                + item_data["image_emb"].unsqueeze(0).to(self.device) * 0.7
            )

            try:
                output = active(
                    text_in, image_in, behavior_vec, return_fusion_weights=True
                )
                if isinstance(output, tuple):
                    logits, fw = output
                else:
                    logits = output
                    fw = None

                probs = torch.softmax(logits, dim=-1)
                expected_rating = (probs * rating_values).sum().item()
                results.append((item_id, expected_rating, probs[0].cpu(), fw))
            except Exception as e:
                logger.debug(f"Error scoring {item_id}: {e}")
                continue

        return results

    # ------------------------------------------------------------------
    def _fallback_recommendations(self, category, top_k, client_id=None):
        """Return items sorted by stored rating when model is unavailable."""
        if client_id is not None:
            ids = self.catalog.client_items.get(client_id, [])
            items = [self.catalog.items_metadata[i] for i in ids if i in self.catalog.items_metadata]
        else:
            items = list(self.catalog.items_metadata.values())
        if category:
            items = [i for i in items if i.get("category") == category]
        items.sort(key=lambda x: x.get("rating", 0), reverse=True)
        items = items[:top_k]

        return {
            "recommendations": [
                {
                    "item_id": it["id"],
                    "title": it.get("title", "Sản phẩm"),
                    "category": it.get("category", "Unknown"),
                    "category_vi": it.get("category_vi", ""),
                    "price": it.get("price", 0),
                    "rating": it.get("rating", 3),
                    "review_count": it.get("review_count", 0),
                    "image_url": it.get("image_url", ""),
                    "predicted_rating": it.get("rating", 3),
                    "confidence": 0.5,
                }
                for it in items
            ],
            "fusion_weights": {"text": 0.33, "image": 0.33, "behavior": 0.34},
            "inference_time_ms": 0,
            "total_candidates": len(items),
            "model_version": "fallback",
        }
