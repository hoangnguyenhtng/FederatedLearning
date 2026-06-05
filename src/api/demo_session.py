"""
Demo shop sessions: federated client login, per-client catalog filter, online personal-head updates.
"""

from __future__ import annotations

import copy
import logging
import pickle
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.api.demo_catalog import (
    client_category,
    client_store_label,
    enrich_product_metadata,
    list_demo_clients,
)

logger = logging.getLogger(__name__)

ACTION_TO_LABEL = {
    "view": 2,
    "click": 3,
    "add_to_cart": 4,
    "cart": 4,
    "purchase": 4,
    "search": 2,
}


class OnlinePersonalizer:
    """FedPer-style local updates: only personal head trains on-device interaction data."""

    def __init__(self, base_model: nn.Module, device: torch.device, lr: float = 0.002):
        self.base_model = base_model
        self.device = device
        self.lr = lr
        self._heads: Dict[int, Dict[str, torch.Tensor]] = {}
        self._buffers: Dict[str, Deque[Dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=64)
        )
        self._stats: Dict[str, Dict[str, Any]] = {}

    def _clone_personal_state(self) -> Dict[str, torch.Tensor]:
        return {
            name: param.detach().cpu().clone()
            for name, param in self.base_model.get_personal_parameters().items()
        }

    def _apply_personal_state(self, model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
        current = model.state_dict()
        for name, param in state.items():
            if name in current:
                current[name] = param.to(self.device)
        model.load_state_dict(current, strict=False)

    def ensure_client(self, client_id: int) -> None:
        if client_id not in self._heads:
            self._heads[client_id] = self._clone_personal_state()

    def get_model_for_inference(self, client_id: int) -> nn.Module:
        self.ensure_client(client_id)
        model = copy.deepcopy(self.base_model)
        model.eval()
        self._apply_personal_state(model, self._heads[client_id])
        return model

    def queue_interaction(
        self,
        session_key: str,
        client_id: int,
        item_id: str,
        action: str,
        *,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
        behavior_vec: torch.Tensor,
    ) -> None:
        label = ACTION_TO_LABEL.get(action, 2)
        self._buffers[session_key].append(
            {
                "client_id": client_id,
                "item_id": item_id,
                "action": action,
                "label": label,
                "text_emb": text_emb.detach().cpu(),
                "image_emb": image_emb.detach().cpu(),
                "behavior": behavior_vec.detach().cpu(),
            }
        )

    def run_local_update(
        self, session_key: str, min_samples: int = 2, steps: int = 3
    ) -> Dict[str, Any]:
        buf = self._buffers[session_key]
        if len(buf) < min_samples:
            return {"updated": False, "reason": "not_enough_interactions", "buffer_size": len(buf)}

        client_id = buf[-1]["client_id"]
        self.ensure_client(client_id)

        model = copy.deepcopy(self.base_model)
        self._apply_personal_state(model, self._heads[client_id])
        model.train()

        for p in model.parameters():
            p.requires_grad = False
        for p in model.personal_head.parameters():
            p.requires_grad = True

        opt = torch.optim.Adam(model.personal_head.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        samples = list(buf)[-min(32, len(buf)) :]

        total_loss = 0.0
        for _ in range(steps):
            epoch_loss = 0.0
            for row in samples:
                text = row["text_emb"].unsqueeze(0).to(self.device)
                image = row["image_emb"].unsqueeze(0).to(self.device)
                behavior = row["behavior"].unsqueeze(0).to(self.device)
                label = torch.tensor([row["label"]], device=self.device, dtype=torch.long)

                opt.zero_grad()
                logits = model(text, image, behavior)
                loss = loss_fn(logits, label)
                loss.backward()
                opt.step()
                epoch_loss += float(loss.item())
            total_loss = epoch_loss / max(len(samples), 1)

        self._heads[client_id] = self._clone_personal_state_from(model)
        self._stats[session_key] = {
            "last_update_at": time.time(),
            "samples_used": len(samples),
            "loss": round(total_loss, 4),
            "client_id": client_id,
            "personal_updates": self._stats.get(session_key, {}).get("personal_updates", 0) + 1,
        }
        return {
            "updated": True,
            "client_id": client_id,
            "samples_used": len(samples),
            "loss": round(total_loss, 4),
            "personal_updates": self._stats[session_key]["personal_updates"],
        }

    def _clone_personal_state_from(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return {
            name: param.detach().cpu().clone()
            for name, param in model.get_personal_parameters().items()
        }

    def get_stats(self, session_key: str) -> Dict[str, Any]:
        return self._stats.get(session_key, {"personal_updates": 0})


class DemoSessionManager:
    """Login as a federated client + customer; filter catalog to that silo."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._users_cache: Dict[int, List[Dict[str, Any]]] = {}

    def list_clients(self) -> List[Dict[str, Any]]:
        return list_demo_clients()

    def _data_path_for_client(self, client_id: int) -> Optional[Path]:
        for sub in ("amazon_2023_processed", "processed/multi_category"):
            p = self.project_root / "data" / sub / f"client_{client_id}" / "data.pkl"
            if p.exists():
                return p
        return None

    def list_users_for_client(self, client_id: int, limit: int = 12) -> List[Dict[str, Any]]:
        if client_id in self._users_cache:
            return self._users_cache[client_id][:limit]

        path = self._data_path_for_client(client_id)
        users: List[Dict[str, Any]] = []
        if path is None:
            for i in range(5):
                uid = f"demo_user_{client_id}_{i}"
                users.append({"user_id": uid, "display_name": f"Khách {i + 1}", "interactions": 10 + i})
            self._users_cache[client_id] = users
            return users[:limit]

        try:
            df = pd.read_pickle(path)
            if "user_id" not in df.columns:
                raise ValueError("no user_id column")
            grp = (
                df.groupby("user_id")
                .agg(interactions=("item_id", "count"))
                .reset_index()
                .sort_values("interactions", ascending=False)
            )
            for _, row in grp.head(max(limit, 20)).iterrows():
                uid = str(row["user_id"])
                users.append(
                    {
                        "user_id": uid,
                        "display_name": self._friendly_user_name(uid, client_id),
                        "interactions": int(row["interactions"]),
                    }
                )
        except Exception as e:
            logger.warning(f"Could not load users for client {client_id}: {e}")
            users = [
                {
                    "user_id": f"user_{client_id}_{i:03d}",
                    "display_name": f"Khách hàng #{i + 1}",
                    "interactions": 20 - i,
                }
                for i in range(5)
            ]

        self._users_cache[client_id] = users
        return users[:limit]

    @staticmethod
    def _friendly_user_name(user_id: str, client_id: int) -> str:
        short = user_id.replace("user_", "").replace("USER_", "")[:12]
        return f"Khách #{short}" if short else f"Khách client {client_id}"

    def create_session(self, client_id: int, user_id: Optional[str] = None) -> Dict[str, Any]:
        if client_id < 0 or client_id > 39:
            raise ValueError("client_id must be 0–39")

        users = self.list_users_for_client(client_id)
        if not user_id:
            user_id = users[0]["user_id"] if users else f"demo_user_{client_id}"

        session_id = f"c{client_id}_{abs(hash(f'{client_id}_{user_id}')) % 10**8}"
        category = client_category(client_id)

        self.sessions[session_id] = {
            "session_id": session_id,
            "client_id": client_id,
            "user_id": user_id,
            "category": category,
            "store_name": client_store_label(client_id),
            "created_at": time.time(),
        }

        return {
            **self.sessions[session_id],
            "user_display": next(
                (u["display_name"] for u in users if u["user_id"] == user_id),
                self._friendly_user_name(user_id, client_id),
            ),
            "available_users": users,
            "fl_note": "Personal head sẽ cập nhật cục bộ khi bạn tương tác — dữ liệu không rời thiết bị.",
        }

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.sessions.get(session_id)
